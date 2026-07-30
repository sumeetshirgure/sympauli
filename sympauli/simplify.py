"""
simplify.py
-----------
Layer 5 of the Symbolic Pauli Heisenberg Evolution Engine.

Two independent concerns live here, both aimed at the same cost centre: the
blanket SymPy call that `PauliSum.simplify()` makes on every coefficient of
every intermediate result during `heisenberg.evolve`.

1.  A *structural* saving.  Most gates in a variational circuit are single
    generator rotations,

        G = exp(-iθ/2 · Q) = cos(θ/2)·I − i·sin(θ/2)·Q,

    and for such a gate the conjugation of one Pauli string P has a closed
    form that never builds the triple product G† P G at all:

        G† P G = P                                   if [P, Q] = 0
        G† P G = cos(θ)·P − i·sin(θ)·(P·Q)           if {P, Q} = 0

    The commuting case is free — the term is copied — and the anticommuting
    case produces exactly two terms instead of the |G|²·|H| = 4|H| products
    the generic path generates and then has to merge and simplify.

    Derivation of the second line.  Write G = a·I + b·Q with a = cos(θ/2) and
    b = −i·sin(θ/2), so that G† = ā·I + b̄·Q since Q is Hermitian.  Expanding,

        G† P G = |a|²·P + āb·PQ + ab̄·QP + |b|²·QPQ,

    and for an anticommuting pair QP = −PQ, hence QPQ = −P, leaving

        G† P G = (|a|² − |b|²)·P + (āb − ab̄)·PQ = cos(θ)·P − i·sin(θ)·PQ.

    Note the operand order: it is P·Q, not Q·P.  The two differ by a sign for
    an anticommuting pair, and the sign matters — with P = X and Q = Z the
    formula above gives cos(θ)·X − sin(θ)·Y, which is the textbook (and
    numerically validated) result for Rz†·X·Rz.

2.  A *coefficient pipeline*.  `simplify_coeffs` extends the vocabulary of
    `PauliSum.simplify` ('trig', 'full', 'expand') with 'cancel' and 'none',
    and adds optional rationalisation of floats, so a caller can pick the
    cheapest rewrite that keeps its coefficients in hand.

----------------------------------------------------------------------
Public API
----------------------------------------------------------------------

    as_rotation(gate)                        -> (PauliString, sp.Expr) | None
    conjugate_by_gate_fast(H, gate, n_qubits, simplify='trig') -> PauliSum
    simplify_coeffs(ps, method='trig', rational=False, prune_tol=1e-14) -> PauliSum
    is_clifford_gate(gate, tol=1e-12)        -> bool
"""

from __future__ import annotations

import sympy as sp

from .pauli_string import PauliString
from .pauli_sum import PauliSum, _PHASE_TO_SYMPY, _is_zero
from .gates import Gate
from .heisenberg import conjugate_by_gate


# ---------------------------------------------------------------------------
# Coefficient simplification pipeline
# ---------------------------------------------------------------------------

_SIMPLIFIERS = {
    "trig":   sp.trigsimp,
    "full":   sp.simplify,
    "expand": sp.expand,
    "cancel": sp.cancel,
    "none":   lambda expr: expr,
}


def simplify_coeffs(
    ps: PauliSum,
    method: str = "trig",
    rational: bool = False,
    prune_tol: float = 1e-14,
) -> PauliSum:
    """
    Simplify every coefficient of `ps` and drop the terms that vanish.

    Parameters
    ----------
    ps        : PauliSum
    method    : 'trig'   → sp.trigsimp  (rotation-gate coefficients)
                'full'   → sp.simplify  (general, slow)
                'expand' → sp.expand    (cheap, algebraic only)
                'cancel' → sp.cancel    (rational functions of the parameters,
                                         e.g. after substituting one symbol)
                'none'   → identity     (prune only)
                An unrecognized name falls back to 'trig', matching
                `PauliSum.simplify`.
    rational  : if True, follow the simplifier with sp.nsimplify(·, rational=True)
                to fold stray floats back into exact rationals.  nsimplify
                raises on plenty of symbolic input, so a failure silently keeps
                the un-rationalized expression.
    prune_tol : a coefficient that `_is_zero` judges zero — symbolically, or
                numerically with |·| < prune_tol — removes its term.

    Returns a new PauliSum; `ps` is untouched.
    """
    fn = _SIMPLIFIERS.get(method, sp.trigsimp)

    result = PauliSum(ps.n)
    for key, coeff in ps._terms.items():
        c = fn(coeff)
        if rational:
            try:
                c = sp.nsimplify(c, rational=True)
            except Exception:
                pass
        if not _is_zero(c, prune_tol):
            result._terms[key] = c
    return result


# ---------------------------------------------------------------------------
# Rotation detection
# ---------------------------------------------------------------------------

def as_rotation(gate: Gate) -> tuple[PauliString, sp.Expr] | None:
    """
    Recognize `gate` as a single-generator Pauli rotation and return its
    generator and angle.

    A rotation gate is stored as the two-term local PauliSum

        cos(θ/2)·I  +  (−i·sin(θ/2))·Q,

    so detection is: exactly two terms, one of them the identity, the identity
    coefficient of the form cos(u) — whence θ = 2u — and the other coefficient
    equal to −i·sin(θ/2).  Both requirements are verified with `_is_zero`
    rather than assumed from the shape, so a two-term gate that merely looks
    like a rotation (`S`, `SX`, `PhaseShift`) is rejected.

    For a numeric angle SymPy has already evaluated cos(θ/2) to a Float and
    the shape is gone; the angle is then recovered as θ = ±2·acos(a), both
    signs being tried because cos is even while sin is odd.

    θ is additionally required to be real.  The closed form of §1 in the module
    docstring uses ā = a, and the exact path's `PauliSum.adjoint()` leaves an
    unevaluated `conjugate(θ)` behind when θ is not known to be real, so the
    two paths would not agree.  Rotations in unknown-reality symbols therefore
    return None and take the generic path.

    Returns (Q, θ) with Q a phase-free PauliString on the gate's *local*
    qubits, or None if the gate is not a rotation of this form.
    """
    ps = gate.pauli_sum
    if len(ps._terms) != 2:
        return None

    n_local = ps.n
    id_key = (0, 0, n_local)
    if id_key not in ps._terms:
        return None

    a = ps._terms[id_key]                                    # cos(θ/2)
    q_key = next(k for k in ps._terms if k != id_key)
    b = ps._terms[q_key]                                     # -i·sin(θ/2)

    candidates: list[sp.Expr] = []
    if isinstance(a, sp.Basic) and a.func is sp.cos:
        candidates.append(2 * a.args[0])
    elif a.is_number and a.is_real:
        try:
            u = sp.acos(a)
        except Exception:
            return None
        candidates.extend([2 * u, -2 * u])

    for θ in candidates:
        if not θ.is_real:
            continue
        if _is_zero(a - sp.cos(θ / 2)) and _is_zero(b + sp.I * sp.sin(θ / 2)):
            x_bits, z_bits, _ = q_key
            return PauliString(x_bits, z_bits, n_local, phase=0), θ

    return None


# ---------------------------------------------------------------------------
# Fast conjugation
# ---------------------------------------------------------------------------

def conjugate_by_gate_fast(
    H: PauliSum,
    gate: Gate,
    n_qubits: int,
    simplify: str | None = "trig",
) -> PauliSum:
    """
    Drop-in faster replacement for `heisenberg.conjugate_by_gate`: compute
    G† · H · G for a gate embedded into an n_qubits system.

    When `as_rotation` recognizes the gate as exp(-iθ/2 · Q), each term of H is
    mapped with the closed form

        G† P G = P                                   if [P, Q] = 0
        G† P G = cos(θ)·P − i·sin(θ)·(P·Q)           if {P, Q} = 0

    which skips the triple product entirely: commuting terms — always including
    the identity — are copied unchanged, and anticommuting terms cost one
    PauliString multiplication.  Any other gate falls back to the exact path,
    so this function is a total replacement, not a special case the caller has
    to dispatch on.

    The result is equal to the exact path's, up to the simplification each
    applies (see `tests/test_simplify.py`, which reduces the difference to
    zero for a battery of gates and observables).

    Parameters mirror `conjugate_by_gate`, except that `simplify` is resolved
    through `simplify_coeffs`, so the extra methods 'cancel' and 'none' work
    here and mean the same thing on both paths.  Note that 'none' still prunes
    zero terms — pruning goes through `_is_zero`, not through the simplifier —
    whereas simplify=None skips the pass entirely.
    """
    rot = as_rotation(gate)
    if rot is None:
        # Take the exact product but do the simplification here, so that the
        # method vocabulary does not depend on which path the gate took.
        result = conjugate_by_gate(H, gate, n_qubits, simplify=None)
        return simplify_coeffs(result, simplify) if simplify else result

    Q_local, θ = rot
    Q = Q_local.embed(list(gate.targets), n_qubits)
    cos_θ = sp.cos(θ)
    minus_i_sin_θ = -sp.I * sp.sin(θ)

    result = PauliSum(n_qubits)
    for (x_bits, z_bits, n), coeff in H._terms.items():
        if n != n_qubits:
            raise ValueError(f"Qubit count mismatch: expected {n_qubits}, got {n}")
        P = PauliString(x_bits, z_bits, n_qubits, phase=0)
        if P.commutes_with(Q):
            result._add_term(x_bits, z_bits, coeff)
        else:
            result._add_term(x_bits, z_bits, cos_θ * coeff)
            PQ = P * Q
            result._add_term(
                PQ.x_bits, PQ.z_bits,
                minus_i_sin_θ * coeff * _PHASE_TO_SYMPY[PQ.phase],
            )

    if simplify:
        result = simplify_coeffs(result, simplify)
    return result


# ---------------------------------------------------------------------------
# Clifford detection
# ---------------------------------------------------------------------------

def is_clifford_gate(gate: Gate, tol: float = 1e-12) -> bool:
    """
    True if `gate` maps every Pauli string to a single Pauli string — i.e. it
    is Clifford, and its conjugation could in principle be run as a tableau
    update instead of an operator product.

    A Clifford is determined by the images of the 2·n generators X_i and Z_i,
    so the test conjugates each of those on the gate's *local* qubits and
    checks that G† P G comes back as one term of unit modulus.  The check is
    numeric: coefficients below `tol` count as absent, and the survivor must
    have |c| = 1 to within `tol`.

    A gate with free symbols cannot be judged numerically and is reported
    False, even though a symbolic rotation is Clifford at special angles —
    `gate_Rx(sp.pi/2)` is Clifford, `gate_Rx(θ)` is not, and only the former
    arrives here with its angle resolved.
    """
    G = gate.pauli_sum
    if G.free_symbols():
        return False

    n_local = G.n
    Gd = G.adjoint()

    for i in range(n_local):
        for x_bits, z_bits in ((1 << i, 0), (0, 1 << i)):
            P = PauliSum.from_pauli(PauliString(x_bits, z_bits, n_local))
            out = Gd * P * G
            magnitudes = []
            for coeff in out._terms.values():
                try:
                    magnitudes.append(abs(complex(sp.expand(coeff).evalf())))
                except (TypeError, ValueError):
                    return False
            surviving = [m for m in magnitudes if m > tol]
            if len(surviving) != 1 or abs(surviving[0] - 1.0) > tol:
                return False

    return True
