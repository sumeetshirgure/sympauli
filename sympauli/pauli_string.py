"""
pauli_string.py
---------------
Layer 1 of the Symbolic Pauli Heisenberg Evolution Engine.

Represents an n-qubit Pauli string using the symplectic (bitmask) representation:
  - x_bits: integer bitmask, bit k set ↔ qubit k has X or Y
  - z_bits: integer bitmask, bit k set ↔ qubit k has Z or Y
  - phase:  integer in {0,1,2,3} representing overall phase {1, i, -1, -i}

Single-qubit Pauli encoding:
  I → (x=0, z=0)
  X → (x=1, z=0)
  Y → (x=1, z=1)
  Z → (x=0, z=1)

Multiplication rule (per qubit):
  XY = iZ, YZ = iX, ZX = iY  (and cyclic / reverse-sign)
  XX = YY = ZZ = I
  Any P·I = I·P = P
"""

from __future__ import annotations
from functools import reduce
from typing import Sequence


# ---------------------------------------------------------------------------
# Phase arithmetic helpers
# ---------------------------------------------------------------------------

def _phase_mul(a: int, b: int) -> int:
    """Multiply two phases encoded as integers mod 4 (0=1, 1=i, 2=-1, 3=-i)."""
    return (a + b) % 4


def _phase_str(phase: int) -> str:
    return {0: "+1", 1: "+i", 2: "-1", 3: "-i"}[phase % 4]


# ---------------------------------------------------------------------------
# Core per-qubit multiplication table
# ---------------------------------------------------------------------------
# Maps (x_a, z_a, x_b, z_b) → (x_out, z_out, delta_phase)
# where delta_phase is added to the running phase (mod 4).
#
# Derivation from Pauli algebra:
#   XY = iZ  → phase +1
#   YX = -iZ → phase +3  (= -i)
#   YZ = iX  → phase +1
#   ZY = -iX → phase +3
#   ZX = iY  → phase +1
#   XZ = -iY → phase +3
#   XX = YY = ZZ = I → phase 0

def _single_qubit_mul(xa: int, za: int, xb: int, zb: int) -> tuple[int, int, int]:
    """
    Multiply single-qubit Paulis A·B.
    Returns (x_out, z_out, delta_phase).
    """
    x_out = xa ^ xb
    z_out = za ^ zb
    # Phase contribution: count how many "anticommuting steps" we accumulate.
    # Using the formula: delta = 2 * (xa & zb) - 2 * (za & xb)  ... mod 4
    # Equivalently, the standard symplectic phase formula:
    delta = (za & xb) - (xa & zb)   # in {-1, 0, 1}
    delta_phase = (2 * delta) % 4   # maps to {0, 2, 2} ... let's use direct table
    # Direct lookup is cleaner and avoids sign confusion:
    # (xa, za, xb, zb): (x_out, z_out, delta_phase)
    table = {
        # I·anything = anything, phase 0
        (0,0,0,0): (0,0,0), (0,0,1,0): (1,0,0), (0,0,0,1): (0,1,0), (0,0,1,1): (1,1,0),
        # X·...
        (1,0,0,0): (1,0,0),   # X·I = X
        (1,0,1,0): (0,0,0),   # X·X = I
        (1,0,0,1): (1,1,3),   # X·Z = -iY  (phase 3 = -i)
        (1,0,1,1): (0,1,1),   # X·Y = iZ   (phase 1 = +i)
        # Z·...
        (0,1,0,0): (0,1,0),   # Z·I = Z
        (0,1,1,0): (1,1,1),   # Z·X = iY   (phase 1 = +i)
        (0,1,0,1): (0,0,0),   # Z·Z = I
        (0,1,1,1): (1,0,3),   # Z·Y = -iX  (phase 3 = -i)
        # Y·...
        (1,1,0,0): (1,1,0),   # Y·I = Y
        (1,1,1,0): (0,1,3),   # Y·X = -iZ  (phase 3 = -i)
        (1,1,0,1): (1,0,1),   # Y·Z = iX   (phase 1 = +i)
        (1,1,1,1): (0,0,0),   # Y·Y = I
    }
    return table[(xa, za, xb, zb)]


# ---------------------------------------------------------------------------
# PauliString class
# ---------------------------------------------------------------------------

class PauliString:
    """
    An n-qubit Pauli string with an overall phase.

    Internally stored as:
      x_bits : int   — bitmask of qubits carrying X or Y
      z_bits : int   — bitmask of qubits carrying Z or Y
      phase  : int   — in {0,1,2,3} encoding {1, i, -1, -i}
      n      : int   — number of qubits

    Qubit ordering: qubit 0 is the least-significant bit.
    """

    __slots__ = ("x_bits", "z_bits", "phase", "n")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, x_bits: int, z_bits: int, n: int, phase: int = 0):
        self.x_bits = x_bits
        self.z_bits = z_bits
        self.n = n
        self.phase = phase % 4

    @classmethod
    def from_string(cls, s: str, phase: int = 0) -> "PauliString":
        """
        Construct from a string like 'IXYZ' or 'XZI'.
        Qubit 0 corresponds to the LAST character (rightmost = least significant).

        Examples:
            PauliString.from_string('XZ')   →  X on qubit 1, Z on qubit 0
            PauliString.from_string('IYX')  →  I on qubit 2, Y on qubit 1, X on qubit 0
        """
        s = s.upper()
        n = len(s)
        x_bits = 0
        z_bits = 0
        for i, ch in enumerate(reversed(s)):  # qubit 0 = rightmost char
            if ch == 'X':
                x_bits |= (1 << i)
            elif ch == 'Z':
                z_bits |= (1 << i)
            elif ch == 'Y':
                x_bits |= (1 << i)
                z_bits |= (1 << i)
            elif ch == 'I':
                pass
            else:
                raise ValueError(f"Unknown Pauli character: '{ch}'")
        return cls(x_bits, z_bits, n, phase)

    @classmethod
    def identity(cls, n: int) -> "PauliString":
        """Return the n-qubit identity I⊗n."""
        return cls(0, 0, n, 0)

    # ------------------------------------------------------------------
    # Single-qubit accessor
    # ------------------------------------------------------------------

    def pauli_at(self, qubit: int) -> str:
        """Return 'I', 'X', 'Y', or 'Z' for the given qubit index."""
        x = (self.x_bits >> qubit) & 1
        z = (self.z_bits >> qubit) & 1
        return {(0,0):'I', (1,0):'X', (1,1):'Y', (0,1):'Z'}[(x, z)]

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------

    def __mul__(self, other: "PauliString") -> "PauliString":
        """Multiply two Pauli strings: self · other."""
        if self.n != other.n:
            raise ValueError(
                f"Qubit count mismatch: {self.n} vs {other.n}. "
                "Embed gates into the full space before multiplying."
            )
        phase = _phase_mul(self.phase, other.phase)
        x_out = 0
        z_out = 0
        for q in range(self.n):
            xa = (self.x_bits >> q) & 1
            za = (self.z_bits >> q) & 1
            xb = (other.x_bits >> q) & 1
            zb = (other.z_bits >> q) & 1
            xr, zr, dp = _single_qubit_mul(xa, za, xb, zb)
            x_out |= (xr << q)
            z_out |= (zr << q)
            phase = _phase_mul(phase, dp)
        return PauliString(x_out, z_out, self.n, phase)

    def __rmul__(self, scalar):
        """Allow scalar * PauliString (returns NotImplemented for non-scalars)."""
        return NotImplemented  # PauliSum handles this

    # ------------------------------------------------------------------
    # Adjoint / conjugate
    # ------------------------------------------------------------------

    def adjoint(self) -> "PauliString":
        """
        Hermitian adjoint: P† = P for X, Y, Z, I (all Hermitian),
        so only the overall phase is conjugated: phase → -phase mod 4.
        """
        return PauliString(self.x_bits, self.z_bits, self.n, (-self.phase) % 4)

    def conjugate(self) -> "PauliString":
        """
        Complex conjugate (transpose in the standard basis).
        Y* = -Y, all others unchanged.
        Extra phase of -1 for each Y qubit (i.e. for each qubit with x=z=1).
        """
        y_count = bin(self.x_bits & self.z_bits).count('1')
        extra_phase = (2 * y_count) % 4  # each Y contributes -1 = phase 2
        return PauliString(self.x_bits, self.z_bits, self.n,
                           _phase_mul(self.phase, extra_phase))

    def transpose(self) -> "PauliString":
        """Transpose = conjugate · adjoint."""
        return self.conjugate().adjoint()

    # ------------------------------------------------------------------
    # Commutativity
    # ------------------------------------------------------------------

    def commutes_with(self, other: "PauliString") -> bool:
        """
        Two Pauli strings commute iff the number of qubits where they
        anti-commute is even.  Anti-commute at qubit q iff exactly one of
        (A has X/Y and B has Z/Y) or (A has Z/Y and B has X/Y) is true.
        Symplectic formula: commutes ↔ popcount(x_A & z_B) + popcount(z_A & x_B) is even.
        """
        anti = bin(self.x_bits & other.z_bits).count('1') + \
               bin(self.z_bits & other.x_bits).count('1')
        return anti % 2 == 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weight(self) -> int:
        """Number of non-identity single-qubit factors."""
        return bin(self.x_bits | self.z_bits).count('1')

    @property
    def is_hermitian(self) -> bool:
        """A Pauli string (with phase) is Hermitian iff phase ∈ {0, 2} (real)."""
        return self.phase % 2 == 0

    @property
    def phase_value(self) -> complex:
        """Return the phase as a Python complex number."""
        return [1, 1j, -1, -1j][self.phase]

    # ------------------------------------------------------------------
    # Tensor product
    # ------------------------------------------------------------------

    def tensor(self, other: "PauliString") -> "PauliString":
        """
        Tensor product self ⊗ other.
        'other' acts on qubits 0..other.n-1, 'self' acts on other.n..self.n+other.n-1.
        """
        n_new = self.n + other.n
        x_new = other.x_bits | (self.x_bits << other.n)
        z_new = other.z_bits | (self.z_bits << other.n)
        phase_new = _phase_mul(self.phase, other.phase)
        return PauliString(x_new, z_new, n_new, phase_new)

    # ------------------------------------------------------------------
    # Embedding into a larger space
    # ------------------------------------------------------------------

    def embed(self, targets: Sequence[int], n_total: int) -> "PauliString":
        """
        Embed this Pauli string (defined on self.n qubits) into an n_total-qubit
        space, placing qubit k of self onto qubit targets[k] of the full space.

        Example:
            X on qubit 0 of a 1-qubit gate, embedded onto qubit 2 of a 4-qubit system:
            PauliString.from_string('X').embed([2], 4)  →  IXII
        """
        if len(targets) != self.n:
            raise ValueError(f"Need {self.n} targets, got {len(targets)}")
        if any(t >= n_total for t in targets):
            raise ValueError("Target qubit index out of range")
        x_new = 0
        z_new = 0
        for local_q, global_q in enumerate(targets):
            if (self.x_bits >> local_q) & 1:
                x_new |= (1 << global_q)
            if (self.z_bits >> local_q) & 1:
                z_new |= (1 << global_q)
        return PauliString(x_new, z_new, n_total, self.phase)

    # ------------------------------------------------------------------
    # Equality and hashing (ignoring phase — for use as dict keys in PauliSum)
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        """Full equality including phase."""
        if not isinstance(other, PauliString):
            return False
        return (self.x_bits == other.x_bits and
                self.z_bits == other.z_bits and
                self.n == other.n and
                self.phase == other.phase)

    def __hash__(self) -> int:
        """Hash including phase (so PauliString can be used as a dict key)."""
        return hash((self.x_bits, self.z_bits, self.n, self.phase))

    def key(self) -> tuple[int, int, int]:
        """
        Hashable key that identifies the Pauli string *without* phase.
        Useful for grouping terms in a PauliSum.
        """
        return (self.x_bits, self.z_bits, self.n)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        phase_s = _phase_str(self.phase)
        label = "".join(self.pauli_at(q) for q in range(self.n - 1, -1, -1))
        return f"PauliString('{label}', phase={phase_s}, n={self.n})"

    def __str__(self) -> str:
        phase_s = _phase_str(self.phase)
        label = "".join(self.pauli_at(q) for q in range(self.n - 1, -1, -1))
        return f"{phase_s}·{label}"

    def label(self) -> str:
        """Return just the Pauli label string, e.g. 'IXYZ', without phase."""
        return "".join(self.pauli_at(q) for q in range(self.n - 1, -1, -1))


# ---------------------------------------------------------------------------
# Convenience: multiply a sequence of PauliStrings
# ---------------------------------------------------------------------------

def pauli_product(*strings: PauliString) -> PauliString:
    """Multiply an arbitrary number of PauliStrings left-to-right."""
    return reduce(lambda a, b: a * b, strings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
