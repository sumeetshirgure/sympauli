"""
pauli_sum.py
------------
Layer 2 of the Symbolic Pauli Heisenberg Evolution Engine.

A PauliSum is a linear combination of Pauli strings with symbolic (SymPy) coefficients:

    S = Σᵢ cᵢ · Pᵢ

where cᵢ are SymPy expressions and Pᵢ are PauliStrings (phase-free, i.e. phase=0).

Design note on phases:
    PauliString multiplication produces phases {1, i, -1, -i} as part of the string.
    In a PauliSum we absorb all phases into the symbolic coefficient and always store
    PauliStrings with phase=0 as keys.  This keeps the dict canonical: there is exactly
    one entry per Pauli label, and all complex structure lives in the SymPy expression.

Supported operations:
    +, -            PauliSum ± PauliSum  (or scalar)
    *               PauliSum * PauliSum  (full operator product)
                    scalar * PauliSum / PauliSum * scalar
    adjoint()       Hermitian conjugate  (conjugate coefficients, each P† = P)
    conjugate()     Complex conjugate of coefficients only
    simplify()      SymPy trigsimp / simplify on each coefficient, prune zeros
    subs()          Substitute numerical or symbolic values for parameters
    to_matrix()     Convert to a 2ⁿ × 2ⁿ NumPy matrix
    commutator()    [A, B] = AB - BA
    anticommutator()  {A, B} = AB + BA
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from typing import Union, Sequence

from .pauli_string import PauliString, _phase_mul

# ---------------------------------------------------------------------------
# Pauli matrices (for to_matrix)
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MATS = {(0, 0): _I2, (1, 0): _X2, (1, 1): _Y2, (0, 1): _Z2}

_PHASE_TO_SYMPY = {0: sp.Integer(1), 1: sp.I, 2: sp.Integer(-1), 3: -sp.I}

Scalar = Union[sp.Expr, int, float, complex]


def _to_sympy(x: Scalar) -> sp.Expr:
    """Coerce a Python scalar or SymPy expression into a SymPy Expr."""
    if isinstance(x, sp.Basic):
        return x
    return sp.sympify(x)


def _is_zero(expr: sp.Expr, tolerance: float = 1e-14) -> bool:
    """
    Check if a SymPy expression is zero.
    Tries exact symbolic check first, then numeric.
    """
    if expr == sp.Integer(0):
        return True
    simplified = sp.simplify(expr)
    if simplified == sp.Integer(0):
        return True
    # Numeric check as fallback
    try:
        val = complex(simplified.evalf())
        return abs(val) < tolerance
    except Exception:
        return False


# ---------------------------------------------------------------------------
# PauliSum
# ---------------------------------------------------------------------------

class PauliSum:
    """
    A symbolic linear combination of n-qubit Pauli strings.

        S = Σᵢ cᵢ · Pᵢ

    Internally stored as:
        _terms : dict[tuple[int,int,int], sp.Expr]
            key   = (x_bits, z_bits, n)  — identifies the Pauli string, phase-free
            value = SymPy expression      — coefficient (absorbs any phase)

    All arithmetic is exact; simplification is on-demand.
    """

    __slots__ = ("_terms", "n")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n: int):
        """Create an empty PauliSum for an n-qubit system."""
        self.n = n
        self._terms: dict[tuple[int, int, int], sp.Expr] = {}

    def _key(self, p: PauliString) -> tuple[int, int, int]:
        if p.n != self.n:
            raise ValueError(f"Qubit count mismatch: expected {self.n}, got {p.n}")
        return (p.x_bits, p.z_bits, self.n)

    def _pauli_from_key(self, key: tuple[int, int, int]) -> PauliString:
        x_bits, z_bits, n = key
        return PauliString(x_bits, z_bits, n, phase=0)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pauli(cls, p: PauliString, coeff: Scalar = 1) -> "PauliSum":
        """
        Wrap a single PauliString into a PauliSum.
        The phase of p is absorbed into the coefficient.
        """
        s = cls(p.n)
        phase_factor = _PHASE_TO_SYMPY[p.phase]
        c = _to_sympy(coeff) * phase_factor
        key = (p.x_bits, p.z_bits, p.n)
        s._terms[key] = c
        return s

    @classmethod
    def from_dict(cls, terms: dict[str, Scalar], n: int) -> "PauliSum":
        """
        Construct from a human-readable dict:
            {'XI': cos(θ/2), 'ZI': -I*sin(θ/2), ...}
        Keys are Pauli label strings; values are SymPy-compatible scalars.
        """
        s = cls(n)
        for label, coeff in terms.items():
            p = PauliString.from_string(label)
            if p.n != n:
                raise ValueError(
                    f"Label '{label}' has {p.n} qubits, expected {n}"
                )
            key = (p.x_bits, p.z_bits, n)
            c = _to_sympy(coeff)
            s._terms[key] = s._terms.get(key, sp.Integer(0)) + c
        return s

    @classmethod
    def zero(cls, n: int) -> "PauliSum":
        """The zero operator on n qubits."""
        return cls(n)

    @classmethod
    def identity(cls, n: int) -> "PauliSum":
        """The identity operator on n qubits (coefficient 1)."""
        return cls.from_pauli(PauliString.identity(n))

    # ------------------------------------------------------------------
    # Internal term manipulation
    # ------------------------------------------------------------------

    def _add_term(self, x_bits: int, z_bits: int, coeff: sp.Expr) -> None:
        """Add a single term in-place (used internally during arithmetic)."""
        key = (x_bits, z_bits, self.n)
        self._terms[key] = self._terms.get(key, sp.Integer(0)) + coeff

    def copy(self) -> "PauliSum":
        s = PauliSum(self.n)
        s._terms = dict(self._terms)
        return s

    # ------------------------------------------------------------------
    # Addition / subtraction
    # ------------------------------------------------------------------

    def __add__(self, other: Union["PauliSum", Scalar]) -> "PauliSum":
        if isinstance(other, PauliSum):
            if self.n != other.n:
                raise ValueError("Qubit count mismatch in addition")
            result = self.copy()
            for key, coeff in other._terms.items():
                x, z, n = key
                result._add_term(x, z, coeff)
            return result
        # scalar: interpret as scalar * Identity
        return self + PauliSum.from_pauli(PauliString.identity(self.n), other)

    def __radd__(self, other: Scalar) -> "PauliSum":
        return self.__add__(other)

    def __sub__(self, other: Union["PauliSum", Scalar]) -> "PauliSum":
        if isinstance(other, PauliSum):
            return self + (-other)
        return self + PauliSum.from_pauli(PauliString.identity(self.n), -_to_sympy(other))

    def __rsub__(self, other: Scalar) -> "PauliSum":
        return (-self).__add__(other)

    def __neg__(self) -> "PauliSum":
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            result._terms[key] = -coeff
        return result

    # ------------------------------------------------------------------
    # Scalar multiplication
    # ------------------------------------------------------------------

    def __mul__(self, other: Union["PauliSum", Scalar]) -> "PauliSum":
        if isinstance(other, PauliSum):
            return self._operator_mul(other)
        # scalar
        c = _to_sympy(other)
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            result._terms[key] = coeff * c
        return result

    def __rmul__(self, other: Scalar) -> "PauliSum":
        if isinstance(other, PauliSum):
            return other._operator_mul(self)
        c = _to_sympy(other)
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            result._terms[key] = c * coeff
        return result

    def __truediv__(self, other: Scalar) -> "PauliSum":
        return self * (sp.Integer(1) / _to_sympy(other))

    # ------------------------------------------------------------------
    # Operator (PauliSum × PauliSum) multiplication
    # ------------------------------------------------------------------

    def _operator_mul(self, other: "PauliSum") -> "PauliSum":
        """
        Compute self · other as operator product.

        For each pair of terms (cᵢ · Pᵢ) and (cⱼ · Qⱼ):
            result += (cᵢ · cⱼ · phase(PᵢQⱼ)) · label(PᵢQⱼ)

        The phase from PauliString multiplication is absorbed into the coefficient.
        """
        if self.n != other.n:
            raise ValueError("Qubit count mismatch in operator multiplication")

        result = PauliSum(self.n)

        for (xa, za, _), ca in self._terms.items():
            pa = PauliString(xa, za, self.n, phase=0)
            for (xb, zb, _), cb in other._terms.items():
                pb = PauliString(xb, zb, self.n, phase=0)
                prod = pa * pb          # PauliString multiplication → phase + label
                phase_factor = _PHASE_TO_SYMPY[prod.phase]
                coeff = ca * cb * phase_factor
                result._add_term(prod.x_bits, prod.z_bits, coeff)

        return result

    # ------------------------------------------------------------------
    # Adjoint / conjugate
    # ------------------------------------------------------------------

    def adjoint(self) -> "PauliSum":
        """
        Hermitian adjoint H†.
        Each Pauli string is self-adjoint (P† = P), so we only conjugate coefficients.
        """
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            result._terms[key] = sp.conjugate(coeff)
        return result

    def conjugate(self) -> "PauliSum":
        """
        Complex conjugate of coefficients (does NOT transpose operators).
        For the operator complex conjugate you'd also conjugate Y → -Y;
        use adjoint() for the Hermitian adjoint instead.
        """
        return self.adjoint()  # same for Pauli strings since P† = P = Pᵀ* for X,Z,I;
                               # for Y: Yᵀ = -Y so Y* = Yᵀ†  = (-Y)† = -Y†= -Y.
                               # But in PauliSum we track Y via x&z bits; the sign
                               # is in the coefficient.  Y* contributes an extra -1
                               # per Y in the string.

    def adjoint_full(self) -> "PauliSum":
        """
        Full Hermitian adjoint including Y sign flip.
        For a Pauli string P with nY Y-factors: P† = (-1)^nY · P  (since Y* = -Y).
        Coefficients are also complex-conjugated.
        """
        result = PauliSum(self.n)
        for (xb, zb, n), coeff in self._terms.items():
            n_Y = bin(xb & zb).count('1')
            sign = sp.Integer(-1)**n_Y
            result._terms[(xb, zb, n)] = sign * sp.conjugate(coeff)
        return result

    # ------------------------------------------------------------------
    # Simplification
    # ------------------------------------------------------------------

    def simplify(self, method: str = "trig") -> "PauliSum":
        """
        Simplify all coefficients and prune zero terms.

        method: 'trig'    → sp.trigsimp  (best for rotation gate coefficients)
                'full'    → sp.simplify  (slower, more general)
                'expand'  → sp.expand    (fast, algebraic expansion only)
        """
        simplifiers = {
            "trig":   sp.trigsimp,
            "full":   sp.simplify,
            "expand": sp.expand,
        }
        fn = simplifiers.get(method, sp.trigsimp)

        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            s = fn(coeff)
            if not _is_zero(s):
                result._terms[key] = s
        return result

    def prune(self, tolerance: float = 1e-14) -> "PauliSum":
        """Remove terms with zero or near-zero coefficients."""
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            if not _is_zero(coeff, tolerance):
                result._terms[key] = coeff
        return result

    # ------------------------------------------------------------------
    # Substitution
    # ------------------------------------------------------------------

    def subs(self, *args, **kwargs) -> "PauliSum":
        """
        Substitute values for symbols in all coefficients.
        Accepts the same arguments as sympy.Expr.subs:
            .subs(theta, 0)
            .subs({theta: sp.pi/4, phi: sp.pi/2})
        """
        result = PauliSum(self.n)
        for key, coeff in self._terms.items():
            result._terms[key] = coeff.subs(*args, **kwargs)
        return result

    def free_symbols(self) -> set:
        """Return the set of all free SymPy symbols appearing in coefficients."""
        syms = set()
        for coeff in self._terms.values():
            syms |= coeff.free_symbols
        return syms

    # ------------------------------------------------------------------
    # Commutator / anticommutator
    # ------------------------------------------------------------------

    def commutator(self, other: "PauliSum") -> "PauliSum":
        """[self, other] = self·other − other·self"""
        return self * other - other * self

    def anticommutator(self, other: "PauliSum") -> "PauliSum":
        """{self, other} = self·other + other·self"""
        return self * other + other * self

    # ------------------------------------------------------------------
    # Numeric matrix representation
    # ------------------------------------------------------------------

    def to_matrix(self, subs: dict | None = None) -> np.ndarray:
        """
        Convert to a 2ⁿ × 2ⁿ NumPy matrix by evaluating all SymPy coefficients
        numerically.

        subs: optional dict of {symbol: value} substitutions to apply first.
        """
        dim = 2 ** self.n
        mat = np.zeros((dim, dim), dtype=complex)
        for (xb, zb, n), coeff in self._terms.items():
            c = coeff
            if subs:
                c = c.subs(subs)
            c_val = complex(c.evalf())
            # Build the Kronecker product for this Pauli string
            # qubit 0 = rightmost in tensor product (least significant bit)
            factors = []
            for q in range(n - 1, -1, -1):   # big-endian: qubit n-1 first
                x = (xb >> q) & 1
                z = (zb >> q) & 1
                factors.append(_PAULI_MATS[(x, z)])
            p_mat = factors[0]
            for f in factors[1:]:
                p_mat = np.kron(p_mat, f)
            mat += c_val * p_mat
        return mat

    # ------------------------------------------------------------------
    # Iteration / inspection
    # ------------------------------------------------------------------

    def terms(self) -> list[tuple[PauliString, sp.Expr]]:
        """Return list of (PauliString, coefficient) pairs, sorted by weight then label."""
        result = []
        for (xb, zb, n), coeff in self._terms.items():
            p = PauliString(xb, zb, n, phase=0)
            result.append((p, coeff))
        result.sort(key=lambda t: (t[0].weight, t[0].label()))
        return result

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self.terms())

    def __eq__(self, other) -> bool:
        if not isinstance(other, PauliSum):
            return False
        diff = (self - other).simplify()
        return len(diff._terms) == 0

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._terms:
            return "PauliSum(0)"
        parts = []
        for p, coeff in self.terms():
            parts.append(f"({coeff})·{p.label()}")
        return " + ".join(parts)

    def __str__(self) -> str:
        return self.__repr__()

    def pretty(self) -> str:
        """Multi-line pretty-printed representation using SymPy's printer."""
        if not self._terms:
            return "0"
        lines = []
        for p, coeff in self.terms():
            coeff_str = sp.pretty(coeff, use_unicode=True)
            lines.append(f"  {coeff_str} · {p.label()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience: embed a PauliSum into a larger space
# ---------------------------------------------------------------------------

def embed_sum(ps: PauliSum, targets: Sequence[int], n_total: int) -> PauliSum:
    """
    Embed a PauliSum defined on len(targets) qubits into an n_total-qubit space.
    Each PauliString in ps is embedded via PauliString.embed().
    """
    result = PauliSum(n_total)
    for (xb, zb, _), coeff in ps._terms.items():
        local_p = PauliString(xb, zb, ps.n, phase=0)
        global_p = local_p.embed(targets, n_total)
        result._add_term(global_p.x_bits, global_p.z_bits, coeff)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
