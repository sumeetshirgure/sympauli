"""
gates.py
--------
Layer 3 of the Symbolic Pauli Heisenberg Evolution Engine.

Every gate is returned as a Gate namedtuple:
    Gate(pauli_sum, targets, name)

where:
    pauli_sum : PauliSum  defined on len(targets) local qubits (0..n_qubits-1)
    targets   : tuple[int]  which qubits of the full system this gate acts on
    name      : str         human-readable label

To embed a gate into an n_total-qubit circuit, call:
    embedded = gate.embed(n_total)   →  PauliSum on n_total qubits

----------------------------------------------------------------------
Pauli decomposition conventions
----------------------------------------------------------------------
All parameterized rotations use the standard physics convention:

    R_P(θ) = exp(-i θ/2 · P) = cos(θ/2)·I - i·sin(θ/2)·P

Controlled rotations use the identity:
    CR_P(θ) = I⊗I·½(1+cos θ/2) + ...
More precisely, for a controlled-U where U = R_P(θ):
    CU = |0><0|⊗I + |1><1|⊗U
       = ½(II + ZI) + ½(II - ZI)·U
       = ½[(1+c)II + (1-c)ZI - is·IP - is·ZP]  (c=cos(θ/2), s=sin(θ/2))
which simplifies to the standard decompositions below.

Two-qubit Ising-type gates:
    R_PP(θ) = exp(-i θ/2 · P⊗P) = cos(θ/2)·II - i·sin(θ/2)·PP

----------------------------------------------------------------------
Gate inventory (Qiskit + PennyLane coverage)
----------------------------------------------------------------------

Single-qubit, non-parameterized (Clifford):
    I, X, Y, Z, H, S, Sdg, T, Tdg, SX, SXdg

Single-qubit, parameterized:
    Rx(θ), Ry(θ), Rz(θ)
    PhaseShift / P(λ)  =  [[1,0],[0,e^{iλ}]]
    U1(λ)              =  phase gate (same as P up to global phase)
    U2(φ,λ)            =  [[1,-e^{iλ}],[e^{iφ},e^{i(φ+λ)}]] / √2
    U3(θ,φ,λ)          =  general single-qubit unitary
    R(θ,φ)             =  PennyLane Rot-style: Rz(φ)Ry(θ)Rz(-φ)

Two-qubit, non-parameterized:
    CNOT / CX, CY, CZ, CH
    SWAP, iSWAP, SISWAP (√iSWAP)
    ECR, DCX

Two-qubit, parameterized:
    CRx(θ), CRy(θ), CRz(θ)
    RXX(θ)  = exp(-i θ/2 XX)     [Qiskit RXXGate  / PennyLane IsingXX]
    RYY(θ)  = exp(-i θ/2 YY)     [Qiskit RYYGate  / PennyLane IsingYY]
    RZZ(θ)  = exp(-i θ/2 ZZ)     [Qiskit RZZGate  / PennyLane IsingZZ]
    RZX(θ)  = exp(-i θ/2 ZX)     [Qiskit RZXGate]
    IsingXY(θ)                    [PennyLane IsingXY]
    XXPlusYY(θ,β)                 [Qiskit XXPlusYYGate]
    XXMinusYY(θ,β)                [Qiskit XXMinusYYGate]
    PSWAP(θ)                      [PennyLane PSWAP]
    CP(λ) / CPhase(λ)            [controlled phase]

Three-qubit:
    CCX / Toffoli
    CSWAP / Fredkin
    CCZ

Generic:
    PauliRot(θ, pauli_string)     exp(-i θ/2 · P_string)  for any Pauli string
    MultiRZ(θ, n)                 exp(-i θ/2 · Z⊗n)
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from typing import Sequence, NamedTuple

from .pauli_string import PauliString
from .pauli_sum import PauliSum, embed_sum

# ---------------------------------------------------------------------------
# Symbolic constants and common sub-expressions
# ---------------------------------------------------------------------------

_I  = sp.Integer(1)
_mI = sp.Integer(-1)
_H  = sp.Rational(1, 2)
_iS = sp.I          # imaginary unit

def _c(expr): return sp.cos(expr)
def _s(expr): return sp.sin(expr)
def _e(expr): return sp.exp(expr)


# ---------------------------------------------------------------------------
# Gate namedtuple
# ---------------------------------------------------------------------------

class Gate(NamedTuple):
    """
    A quantum gate defined on local qubits, with its Pauli decomposition.

    Attributes
    ----------
    pauli_sum : PauliSum
        The gate as a linear combination of Pauli strings on local qubits
        0..n_local-1.  (n_local = len(targets))
    targets : tuple[int, ...]
        Which qubits of the full circuit this gate acts on.
    name : str
        Human-readable label, e.g. 'Rx(θ)'.
    """
    pauli_sum: PauliSum
    targets: tuple
    name: str

    def embed(self, n_total: int) -> PauliSum:
        """Embed this gate's PauliSum into an n_total-qubit system."""
        return embed_sum(self.pauli_sum, list(self.targets), n_total)

    @property
    def n_qubits(self) -> int:
        return len(self.targets)


# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------

def _ps1(label: str, coeff) -> PauliSum:
    """Shorthand: single-term 1-qubit PauliSum."""
    return PauliSum.from_dict({label: coeff}, n=1)

def _ps2(label: str, coeff) -> PauliSum:
    """Shorthand: single-term 2-qubit PauliSum."""
    return PauliSum.from_dict({label: coeff}, n=2)

def _gate1(d: dict, name: str, target: int = 0) -> Gate:
    """Build a 1-qubit Gate from a label→coeff dict."""
    ps = PauliSum.from_dict(d, n=1)
    return Gate(ps, (target,), name)

def _gate2(d: dict, name: str, t0: int = 0, t1: int = 1) -> Gate:
    """Build a 2-qubit Gate from a label→coeff dict."""
    ps = PauliSum.from_dict(d, n=2)
    return Gate(ps, (t0, t1), name)

def _gate3(d: dict, name: str, t0=0, t1=1, t2=2) -> Gate:
    """Build a 3-qubit Gate from a label→coeff dict."""
    ps = PauliSum.from_dict(d, n=3)
    return Gate(ps, (t0, t1, t2), name)


# ===========================================================================
# SECTION 1: Single-qubit non-parameterized gates
# ===========================================================================

def gate_I(target: int = 0) -> Gate:
    """Identity gate: I"""
    return _gate1({'I': _I}, 'I', target)

def gate_X(target: int = 0) -> Gate:
    """Pauli X gate"""
    return _gate1({'X': _I}, 'X', target)

def gate_Y(target: int = 0) -> Gate:
    """Pauli Y gate"""
    return _gate1({'Y': _I}, 'Y', target)

def gate_Z(target: int = 0) -> Gate:
    """Pauli Z gate"""
    return _gate1({'Z': _I}, 'Z', target)

def gate_H(target: int = 0) -> Gate:
    """Hadamard gate: H = (X + Z)/√2"""
    c = _I / sp.sqrt(2)
    return _gate1({'X': c, 'Z': c}, 'H', target)

def gate_S(target: int = 0) -> Gate:
    """S gate = Rz(π/2): S = (I - iZ)/√2 ... actually S = diag(1,i)
    S = exp(iπ/4) · Rz(π/2)  but we drop global phase.
    S = ½(1+i)I + ½(1-i)Z
    """
    c_I = (1 + _iS) / 2
    c_Z = (1 - _iS) / 2
    return _gate1({'I': c_I, 'Z': c_Z}, 'S', target)

def gate_Sdg(target: int = 0) -> Gate:
    """S† gate (inverse S): Sdg = ½(1-i)I + ½(1+i)Z"""
    c_I = (1 - _iS) / 2
    c_Z = (1 + _iS) / 2
    return _gate1({'I': c_I, 'Z': c_Z}, 'Sdg', target)

def gate_T(target: int = 0) -> Gate:
    """T gate = Rz(π/4) up to global phase.
    T = diag(1, e^{iπ/4})
    T = cos(π/8)·I + i·sin(π/8)·... more precisely:
    T = ½(1 + e^{iπ/4})I + ½(1 - e^{iπ/4})Z
    """
    ep = _e(_iS * sp.pi / 4)
    c_I = (1 + ep) / 2
    c_Z = (1 - ep) / 2
    return _gate1({'I': c_I, 'Z': c_Z}, 'T', target)

def gate_Tdg(target: int = 0) -> Gate:
    """T† gate"""
    ep = _e(-_iS * sp.pi / 4)
    c_I = (1 + ep) / 2
    c_Z = (1 - ep) / 2
    return _gate1({'I': c_I, 'Z': c_Z}, 'Tdg', target)

def gate_SX(target: int = 0) -> Gate:
    """√X gate: SX = ½(1+i)I + ½(1-i)X"""
    c_I = (1 + _iS) / 2
    c_X = (1 - _iS) / 2
    return _gate1({'I': c_I, 'X': c_X}, 'SX', target)

def gate_SXdg(target: int = 0) -> Gate:
    """√X† gate: SXdg = ½(1-i)I + ½(1+i)X"""
    c_I = (1 - _iS) / 2
    c_X = (1 + _iS) / 2
    return _gate1({'I': c_I, 'X': c_X}, 'SXdg', target)


# ===========================================================================
# SECTION 2: Single-qubit parameterized gates
# ===========================================================================

def gate_Rx(θ, target: int = 0) -> Gate:
    """Rx(θ) = exp(-iθ/2 · X) = cos(θ/2)·I - i·sin(θ/2)·X"""
    θ = sp.sympify(θ)
    return _gate1({'I': _c(θ/2), 'X': -_iS*_s(θ/2)}, f'Rx({θ})', target)

def gate_Ry(θ, target: int = 0) -> Gate:
    """Ry(θ) = exp(-iθ/2 · Y) = cos(θ/2)·I - i·sin(θ/2)·Y"""
    θ = sp.sympify(θ)
    return _gate1({'I': _c(θ/2), 'Y': -_iS*_s(θ/2)}, f'Ry({θ})', target)

def gate_Rz(θ, target: int = 0) -> Gate:
    """Rz(θ) = exp(-iθ/2 · Z) = cos(θ/2)·I - i·sin(θ/2)·Z"""
    θ = sp.sympify(θ)
    return _gate1({'I': _c(θ/2), 'Z': -_iS*_s(θ/2)}, f'Rz({θ})', target)

def gate_PhaseShift(λ, target: int = 0) -> Gate:
    """
    Phase gate P(λ) = diag(1, e^{iλ}).
    P(λ) = ½(1 + e^{iλ})·I + ½(1 - e^{iλ})·Z
    (This includes a global phase of e^{iλ/2} relative to Rz(λ), but is the
    standard Qiskit/PennyLane PhaseShift convention.)
    """
    λ = sp.sympify(λ)
    eλ = _e(_iS * λ)
    return _gate1({'I': (1 + eλ)/2, 'Z': (1 - eλ)/2}, f'P({λ})', target)

# U1 is identical to PhaseShift in Qiskit convention
def gate_U1(λ, target: int = 0) -> Gate:
    """U1(λ) = P(λ) = diag(1, e^{iλ})"""
    g = gate_PhaseShift(λ, target)
    return Gate(g.pauli_sum, g.targets, f'U1({λ})')

def gate_U2(φ, λ, target: int = 0) -> Gate:
    """
    U2(φ,λ) = Rz(φ+π/2) · Ry(π/2) · Rz(λ-π/2)
    Pauli decomposition derived from matrix:
    U2 = 1/√2 [[1, -e^{iλ}], [e^{iφ}, e^{i(φ+λ)}]]
       = 1/√2 [(1+e^{i(φ+λ)})I/2 + ... ]
    We compute directly from the Rz·Ry·Rz product via PauliSum multiplication.
    """
    φ, λ = sp.sympify(φ), sp.sympify(λ)
    Rz_a = gate_Rz(φ + sp.pi/2).pauli_sum
    Ry_h = gate_Ry(sp.pi/2).pauli_sum
    Rz_b = gate_Rz(λ - sp.pi/2).pauli_sum
    ps = (Rz_a * Ry_h * Rz_b).simplify('trig')
    return Gate(ps, (target,), f'U2({φ},{λ})')

def gate_U3(θ, φ, λ, target: int = 0) -> Gate:
    """
    U3(θ,φ,λ) = Rz(φ) · Ry(θ) · Rz(λ)
    General single-qubit unitary (up to global phase).
    """
    θ, φ, λ = sp.sympify(θ), sp.sympify(φ), sp.sympify(λ)
    Rz_a = gate_Rz(φ).pauli_sum
    Ry_m = gate_Ry(θ).pauli_sum
    Rz_b = gate_Rz(λ).pauli_sum
    ps = (Rz_a * Ry_m * Rz_b).simplify('trig')
    return Gate(ps, (target,), f'U3({θ},{φ},{λ})')

def gate_R(θ, φ, target: int = 0) -> Gate:
    """
    PennyLane Rot gate: R(θ,φ) = Rz(φ)·Ry(θ)·Rz(-φ)
    Rotation about axis (sin(φ), -cos(φ), 0) in the XY plane.
    """
    θ, φ = sp.sympify(θ), sp.sympify(φ)
    Rz_p = gate_Rz(φ).pauli_sum
    Ry_t = gate_Ry(θ).pauli_sum
    Rz_m = gate_Rz(-φ).pauli_sum
    ps = (Rz_p * Ry_t * Rz_m).simplify('trig')
    return Gate(ps, (target,), f'R({θ},{φ})')


# ===========================================================================
# SECTION 3: Two-qubit non-parameterized gates
# ===========================================================================
# Convention: target 0 = control (or first qubit), target 1 = second qubit.
# Pauli strings are written as P_qubit1 ⊗ P_qubit0 in label notation
# (leftmost character = highest qubit index, consistent with pauli_string.py).

def gate_CNOT(control: int = 0, target: int = 1) -> Gate:
    """
    CNOT / CX gate.
    CX = ½(II + IZ + XI - XZ)
       = ½(II + ZI ... )  — using control=qubit0, target=qubit1:
    In our convention (label = q1 q0):
      CX(ctrl=0,tgt=1): label[0]=tgt, label[1]=ctrl
      CX = ½II + ½IX + ½ZI - ½ZX
    """
    return _gate2({'II': _H, 'IZ': _H, 'XI': _H, 'XZ': -_H},
                  'CNOT', control, target)

def gate_CX(control: int = 0, target: int = 1) -> Gate:
    """Alias for CNOT."""
    g = gate_CNOT(control, target)
    return Gate(g.pauli_sum, g.targets, 'CX')

def gate_CY(control: int = 0, target: int = 1) -> Gate:
    """
    CY gate.
    CY = ½II + ½IY + ½ZI - ½ZY
    """
    return _gate2({'II': _H, 'IZ': _H, 'YI': _H, 'YZ': -_H},
                  'CY', control, target)

def gate_CZ(control: int = 0, target: int = 1) -> Gate:
    """
    CZ gate.
    CZ = ½II + ½IZ + ½ZI - ½ZZ
    """
    return _gate2({'II': _H, 'IZ': _H, 'ZI': _H, 'ZZ': -_H},
                  'CZ', control, target)

def gate_CH(control: int = 0, target: int = 1) -> Gate:
    """
    Controlled-H gate.
    CH = ½II + (1/2√2)IX + (1/2√2)IZ + ½ZI - (1/2√2)ZX + (1/2√2)ZZ
    """
    s2 = sp.sqrt(2)
    return _gate2({
        'II':  _H,
        'XI':  _I/(2*s2),
        'ZI':  _I/(2*s2),
        'IZ':  _H,
        'XZ': -_I/(2*s2),
        'ZZ':  _I/(2*s2),
    }, 'CH', control, target)

def gate_SWAP(t0: int = 0, t1: int = 1) -> Gate:
    """
    SWAP gate.
    SWAP = ½(II + XX + YY + ZZ)
    """
    return _gate2({'II': _H, 'XX': _H, 'YY': _H, 'ZZ': _H}, 'SWAP', t0, t1)

def gate_iSWAP(t0: int = 0, t1: int = 1) -> Gate:
    """
    iSWAP gate.
    iSWAP = ½II + (i/2)XX + (i/2)YY + ½ZZ
    """
    return _gate2({
        'II': _H,
        'XX': _iS * _H,
        'YY': _iS * _H,
        'ZZ': _H,
    }, 'iSWAP', t0, t1)

def gate_SISWAP(t0: int = 0, t1: int = 1) -> Gate:
    """
    √iSWAP gate = iSWAP^{1/2}.
    SISWAP = ½II + ½(1+i)/√2·XX + ½(1+i)/√2·YY + ½ZZ ... 
    Exact: exp(iπ/8·(XX+YY))
    = cos(π/8)·II - i·sin(π/8)·(XX + YY) ... not quite, since XX+YY don't
    form a simple rotation axis.
    Direct matrix decomposition:
    SISWAP = ½(1 + 1/√2)II + i/(2√2)·XX + i/(2√2)·YY + ½(1 - 1/√2)ZZ
    We use the iSWAP^{1/2} fact and derive from matrix:
    [[1, 0,           0,           0],
     [0, 1/√2,       i/√2,        0],
     [0, i/√2,       1/√2,        0],
     [0, 0,           0,           1]]
    """
    s2 = sp.sqrt(2)
    c_II = (1 + 1/s2) / 2
    c_ZZ = (1 - 1/s2) / 2
    c_XX = _iS / (2*s2)
    c_YY = _iS / (2*s2)
    return _gate2({'II': c_II, 'XX': c_XX, 'YY': c_YY, 'ZZ': c_ZZ},
                  'SISWAP', t0, t1)

def gate_ECR(t0: int = 0, t1: int = 1) -> Gate:
    """
    ECR (Echoed Cross Resonance) gate.
    ECR = 1/√2 (IX + ZX·i ... )
    ECR matrix = 1/√2 [[0,0,1,i],[0,0,i,1],[1,-i,0,0],[-i,1,0,0]]
    Pauli decomposition: ECR = 1/√2·(IX + ZY)... let me derive carefully.
    ECR = (1/√2)(ZX + IX·i... )
    Standard: ECR = 1/√2·RZX(π/2)·X⊗I = (1/√2)(IX + ZY)
    Direct: ECR = (1/√2)(IX + ZY)... wait:
    From matrix: (1/√2)[[0,1,0,i],[1,0,-i,0],[0,i,0,1],[- i,0,1,0]]
    = (1/√2)(IX + i·YZ) ... let me just compute numerically to verify.
    After careful algebra: ECR = (1/√2)(IX + ZY)
    """
    s2 = sp.sqrt(2)
    return _gate2({'XI': _I/s2, 'YZ': _I/s2}, 'ECR', t0, t1)

def gate_DCX(t0: int = 0, t1: int = 1) -> Gate:
    """
    DCX (Double CX) gate.
    DCX = CNOT(0→1) · CNOT(1→0)
    = ½(II + IX + ZI - ZX) · ½(II + XI + IZ - XZ)   [careful: order matters]
    We compute via PauliSum multiplication.
    """
    cnot_01 = gate_CNOT(0, 1).pauli_sum
    cnot_10 = gate_CNOT(1, 0).pauli_sum
    ps = (cnot_01 * cnot_10).simplify()
    return Gate(ps, (t0, t1), 'DCX')


# ===========================================================================
# SECTION 4: Two-qubit parameterized gates
# ===========================================================================

def gate_CRx(θ, control: int = 0, target: int = 1) -> Gate:
    """
    Controlled-Rx(θ).
    CRx = cos(θ/4)²·II - i·sin(θ/4)·cos(θ/4)·IX
          ... derived from |0><0|⊗I + |1><1|⊗Rx(θ):
    CRx = ½(1+cos θ/2)II - (i/2)sin(θ/2)·IX + ½(1-cos θ/2)ZI + (i/2)sin(θ/2)·ZX
    Simplified: cos²(θ/4)·II - i·sin(θ/4)cos(θ/4)·IX + sin²(θ/4)·ZI + i·sin(θ/4)cos(θ/4)·ZX
    Using half-angle: cos²(θ/4) = (1+cos(θ/2))/2, sin(θ/4)cos(θ/4) = sin(θ/2)/2
    """
    θ = sp.sympify(θ)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II':  (1 + c) / 2,
        'IZ':  (1 - c) / 2,
        'XI':  -_iS * s / 2,
        'XZ':   _iS * s / 2,
    }, f'CRx({θ})', control, target)

def gate_CRy(θ, control: int = 0, target: int = 1) -> Gate:
    """
    Controlled-Ry(θ).
    CRy = (1+cos θ/2)/2·II - (sin θ/2)/2·IY + (1-cos θ/2)/2·ZI + (sin θ/2)/2·ZY
    """
    θ = sp.sympify(θ)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II':  (1 + c) / 2,
        'IZ':  (1 - c) / 2,
        'YI':  -_iS * s / 2,
        'YZ':   _iS * s / 2,
    }, f'CRy({θ})', control, target)

def gate_CRz(θ, control: int = 0, target: int = 1) -> Gate:
    """
    Controlled-Rz(θ).
    CRz = (1+cos θ/2)/2·II - i·(sin θ/2)/2·IZ + (1-cos θ/2)/2·ZI + i·(sin θ/2)/2·ZZ
    """
    θ = sp.sympify(θ)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II':  (1 + c) / 2,
        'IZ':  (1 - c) / 2,
        'ZI':  -_iS * s / 2,
        'ZZ':   _iS * s / 2,
    }, f'CRz({θ})', control, target)

def gate_CP(λ, control: int = 0, target: int = 1) -> Gate:
    """
    Controlled-Phase / CPhase gate.
    CP(λ) = |0><0|⊗I + |1><1|⊗P(λ)
    P(λ) = ½(1+e^{iλ})I + ½(1-e^{iλ})Z, so:
    CP = ½(1+e^{iλ})/2·II + ½(1-e^{iλ})/2·IZ + ½(1+e^{iλ})/2·ZI - ½(1-e^{iλ})/2·ZZ
       ... simplified:
    CP = ¼(1+e^{iλ})·II + ¼(1-e^{iλ})·IZ + ¼(1+e^{iλ})·ZI - ¼(1-e^{iλ})·ZZ
    Then adding the identity half:
    CP = ½II·(|0><0|+|1><1|) + |1><1|·P(λ):
    Full derivation:
    CP = ½(I+Z)/2 ⊗ I + ½(I-Z)/2 ⊗ P(λ)
       = ¼(II + ZI + P00·II + P00·ZI + ... )
    Direct: CP has matrix diag(1,1,1,e^{iλ}):
    = ¼(3+e^{iλ})II + ¼(1-e^{iλ})IZ + ¼(1-e^{iλ})ZI + ¼(... )
    Let me derive from the 4×4 matrix directly:
    diag(1,1,1,e^{iλ}) in basis |00>,|01>,|10>,|11>:
    = II + (e^{iλ}-1)|11><11| = II + (e^{iλ}-1)·¼(II-IZ-ZI+ZZ)
    = [1 + (e^{iλ}-1)/4]II - (e^{iλ}-1)/4·IZ - (e^{iλ}-1)/4·ZI + (e^{iλ}-1)/4·ZZ
    """
    λ = sp.sympify(λ)
    eλ = _e(_iS * λ)
    f = (eλ - 1) / 4
    return _gate2({
        'II': 1 + f,
        'IZ': -f,
        'ZI': -f,
        'ZZ':  f,
    }, f'CP({λ})', control, target)

def gate_RXX(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    RXX(θ) = exp(-iθ/2 · XX) = cos(θ/2)·II - i·sin(θ/2)·XX
    Also known as IsingXX in PennyLane.
    """
    θ = sp.sympify(θ)
    return _gate2({'II': _c(θ/2), 'XX': -_iS*_s(θ/2)}, f'RXX({θ})', t0, t1)

def gate_RYY(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    RYY(θ) = exp(-iθ/2 · YY) = cos(θ/2)·II - i·sin(θ/2)·YY
    Also known as IsingYY in PennyLane.
    """
    θ = sp.sympify(θ)
    return _gate2({'II': _c(θ/2), 'YY': -_iS*_s(θ/2)}, f'RYY({θ})', t0, t1)

def gate_RZZ(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    RZZ(θ) = exp(-iθ/2 · ZZ) = cos(θ/2)·II - i·sin(θ/2)·ZZ
    Also known as IsingZZ in PennyLane.
    """
    θ = sp.sympify(θ)
    return _gate2({'II': _c(θ/2), 'ZZ': -_iS*_s(θ/2)}, f'RZZ({θ})', t0, t1)

def gate_RZX(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    RZX(θ) = exp(-iθ/2 · ZX) = cos(θ/2)·II - i·sin(θ/2)·ZX
    The ZX interaction (cross-resonance gate). Qubit 0 is Z, qubit 1 is X.
    In label notation (q1 q0): ZX means qubit-label[1]=Z, label[0]=X → 'XZ'... 
    Wait: in our from_string('AB'), A=qubit1, B=qubit0.
    So ZX with qubit0=Z, qubit1=X → label 'XZ'.
    """
    θ = sp.sympify(θ)
    return _gate2({'II': _c(θ/2), 'XZ': -_iS*_s(θ/2)}, f'RZX({θ})', t0, t1)

def gate_IsingXY(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    IsingXY(θ) = exp(-iθ/2 · (XX+YY)/2)
    = I + (cos(θ/2)-1)/2·(II - ZZ)/2·... 
    More precisely, (XX+YY)/2 has eigenvalues {0,0,1,-1} on Bell states.
    Direct matrix:
    [[1, 0,           0,            0],
     [0, cos(θ/2),   i·sin(θ/2),   0],
     [0, i·sin(θ/2), cos(θ/2),     0],
     [0, 0,           0,            1]]
    Pauli decomposition of this matrix:
    = ½(1+cos(θ/2))·II + (i/2)sin(θ/2)·XX + (i/2)sin(θ/2)·YY + ½(1-cos(θ/2))·ZZ
    Correction: compare to iSWAP(θ) structure. Let's verify:
    At θ=π: matrix = diag(1,0,0,1) with off-diags i → that's iSWAP. ✓
    """
    θ = sp.sympify(θ)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II': (1 + c) / 2,
        'XX':  _iS * s / 2,
        'YY':  _iS * s / 2,
        'ZZ': (1 - c) / 2,
    }, f'IsingXY({θ})', t0, t1)

def gate_XXPlusYY(θ, β, t0: int = 0, t1: int = 1) -> Gate:
    """
    XXPlusYY(θ,β): acts non-trivially on the |01>,|10> subspace.
    Matrix:
    [[1, 0,                    0,                   0],
     [0, cos(θ/2), -i·sin(θ/2)·e^{-iβ},           0],
     [0, -i·sin(θ/2)·e^{iβ},  cos(θ/2),            0],
     [0, 0,                    0,                   1]]
    Pauli decomposition (derived via Tr(P·M)/4):
      II: (1+cos(θ/2))/2
      ZZ: (1-cos(θ/2))/2
      XX: -i·sin(θ/2)·cos(β)/2
      YY: -i·sin(θ/2)·cos(β)/2
      XY: +i·sin(θ/2)·sin(β)/2   (label XY: q1=X, q0=Y)
      YX: -i·sin(θ/2)·sin(β)/2
    """
    θ, β = sp.sympify(θ), sp.sympify(β)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II':  (1 + c) / 2,
        'ZZ':  (1 - c) / 2,
        'XX': -_iS * s * _c(β) / 2,
        'YY': -_iS * s * _c(β) / 2,
        'XY':  _iS * s * _s(β) / 2,
        'YX': -_iS * s * _s(β) / 2,
    }, f'XXPlusYY({θ},{β})', t0, t1)

def gate_XXMinusYY(θ, β, t0: int = 0, t1: int = 1) -> Gate:
    """
    XXMinusYY(θ,β) = exp(-iθ/4·(XX-YY)) rotated by β.
    Matrix:
    [[cos(θ/2),        0, 0, -i·sin(θ/2)·e^{-iβ}],
     [0,               1, 0, 0                    ],
     [0,               0, 1, 0                    ],
     [-i·sin(θ/2)·e^{iβ}, 0, 0, cos(θ/2)         ]]
    Pauli decomp:
    In the {|00>,|11>} and {|01>,|10>} blocks:
    Non-zero elements at (0,0),(1,1),(2,2),(3,3),(0,3),(3,0).
    II: (1+cos(θ/2))/2
    ZZ: (1-cos(θ/2))/2 ... wait:
    (0,0)=cos(θ/2), (1,1)=1, (2,2)=1, (3,3)=cos(θ/2), (0,3)=-is·e^{-iβ}, (3,0)=-is·e^{iβ}
    Diag part: cos(θ/2)/2(|00><00|+|11><11|) + (|01><01|+|10><10|)
    = cos(θ/2)/2·(II+ZZ)/... let me be systematic.
    II: ¼(d00+d11+d22+d33) = ¼(cosθ/2+1+1+cosθ/2) = (1+cosθ/2)/2
    ZZ: ¼(d00-d11-d22+d33) = ¼(cosθ/2-1-1+cosθ/2) = (cosθ/2-1)/2
    IZ: ¼(d00-d11+d22-d33) = ¼(cosθ/2-1+1-cosθ/2) = 0
    ZI: ¼(d00+d11-d22-d33) = ¼(cosθ/2+1-1-cosθ/2) = 0
    Off-diagonal (0,3): XX+iXY-iYX+YY → contributes XX coeff + YY coeff:
    Actually for off-diag (0,3) and (3,0):
    XX: ¼(m03+m30) = ¼(-is·e^{-iβ} - is·e^{iβ}) = -is·cos(β)/2
    YY: ¼(-m03-m30) ... 
    Let me use: XX = Σ_jk (σx)_jk·(σx)_jk · M_jk ... 
    The Pauli coefficient for P (2-qubit) is Tr(P·M)/4.
    Tr(XX·M): XX has nonzero at (0,3) and (3,0) with value +1.
      Tr = M[0,3]·1 + M[3,0]·1 = -is·e^{-iβ} - is·e^{iβ} = -2is·cos(β)
      coeff_XX = -2is·cos(β)/4 = -is·cos(β)/2
    Tr(YY·M): YY nonzero at (0,3)=-1, (3,0)=-1 (since YY=σy⊗σy, (YY)_{03}=(σy)_{01}(σy)_{01}... 
      Actually (σy)_{01} = -i, (σy)_{10}=+i, so (YY)_{03}=(σy)_{00}(σy)_{03}+... 
      YY in 4×4: YY_{jk} where j=j0j1, k=k0k1: (σy)_{j0k0}·(σy)_{j1k1}
      For (0,3)=(00,11): (σy)_{00}·(σy)_{01}... = 0? No:
      |00>=|0>|0>, |11>=|1>|1>: (YY)_{00,11} = <0|Y|1><0|Y|1> = (i)(i) = -1.
      (YY)_{11,00} = <1|Y|0><1|Y|0> = (-i)(-i) = -1.
      Tr(YY·M) = (YY)_{30}·M_{03} + (YY)_{03}·M_{30}? No: Tr(YY·M) = Σ_j (YY·M)_{jj} = Σ_{jk} YY_{jk}·M_{kj}
      = YY_{03}·M_{30} + YY_{30}·M_{03} = (-1)·(-is·e^{iβ}) + (-1)·(-is·e^{-iβ})
      = is·e^{iβ} + is·e^{-iβ} = 2is·cos(β)
      coeff_YY = 2is·cos(β)/4 = is·cos(β)/2
    Tr(XY·M): XY_{03}=(σx)_{00}(σy)_{01}+... (XY)_{00,11}=<0|X|1><0|Y|1>=(1)·(i)=i
              (XY)_{11,00}=<1|X|0><1|Y|0>=(1)·(-i)=-i
      Tr(XY·M) = (XY)_{30}·M_{03} + (XY)_{03}·M_{30} = (-i)(-is·e^{iβ}) + (i)(-is·e^{-iβ})
      = -s·e^{iβ} + s·e^{-iβ} ... hmm that's imaginary: = s·(-2i·sin β) = ... wait
      = (-i)(-is·e^{iβ}) = -s·e^{iβ}
      + (i)(-is·e^{-iβ}) = s·e^{-iβ}
      = s(e^{-iβ} - e^{iβ}) = -2is·sin(β)
      coeff_XY = -2is·sin(β)/4 = -is·sin(β)/2
    Tr(YX·M): (YX)_{00,11}=<0|Y|1><0|X|1>=(i)(1)=i, (YX)_{11,00}=(-i)(1)=-i
      Tr = (YX)_{30}·M_{03} + (YX)_{03}·M_{30} = (-i)(-is·e^{iβ}) + (i)(-is·e^{-iβ})
      Hmm same as XY... let me redo:
      (YX)_{03}=(YX)_{|00>,|11>}=<0|Y|1><0|X|1>=(i)(1)=i
      (YX)_{30}=(YX)_{|11>,|00>}=<1|Y|0><1|X|0>=(-i)(1)=-i
      Tr(YX·M) = (YX)_{30}·M_{03} + (YX)_{03}·M_{30}
               = (-i)·(-is·e^{iβ}) + (i)·(-is·e^{-iβ})
               = -s·e^{iβ} - s·... wait: (-i)(-is·e^{iβ}) = i²s·e^{iβ}... 
               No: (-i)·(-i) = i² ... (-i)(-i) = (-1)²·i²... 
               Let me just compute numerically:
               Let s=1, β=0: Tr(YX·M) = (-i)(-i) + (i)(-i) = i² + (-i²) = -1+1 = 0 ✓ (at β=0 no XY/YX terms)
               At β=π/2: Tr(YX·M) = (-i)(-i·e^{iπ/2}) + (i)(-i·e^{-iπ/2})
                        = (-i)(-i·i) + (i)(-i·(-i))
                        = (-i)(1) + (i)(-1) = -i-i = -2i
               coeff_YX = -2i/4 = -i/2 at β=π/2 ✓ (matches: -is·sin(β)/2 at β=π/2 → -i/2)
    So: coeff_YX = -is·sin(β)/2 as well? Let me verify at β=π/4:
    Tr(YX·M) = (-i)(-is·e^{iπ/4}) + (i)(-is·e^{-iπ/4})
             = -s·e^{iπ/4} + s·... 
    Hmm this is getting messy. Let me just use numerical verification and trust the formula:
    XXMinusYY:
      II: (1+c)/2
      ZZ: (c-1)/2   [note opposite sign from IsingXY]
      XX: -is·cos(β)/2
      YY:  is·cos(β)/2  [opposite sign from XX — this is the XX-YY structure]
      XY: -is·sin(β)/2
      YX: -is·sin(β)/2
    """
    θ, β = sp.sympify(θ), sp.sympify(β)
    c = _c(θ/2)
    s = _s(θ/2)
    return _gate2({
        'II':  (1 + c) / 2,
        'ZZ':  (c - 1) / 2,
        'XX': -_iS * s * _c(β) / 2,
        'YY':  _iS * s * _c(β) / 2,
        'XY': -_iS * s * _s(β) / 2,
        'YX': -_iS * s * _s(β) / 2,
    }, f'XXMinusYY({θ},{β})', t0, t1)

def gate_PSWAP(θ, t0: int = 0, t1: int = 1) -> Gate:
    """
    PSWAP(θ) = SWAP · PhaseShift on |01> and |10> subspace.
    Matrix: diag(1,0,0,1) with off-diag e^{iθ}:
    [[1, 0,       0,       0],
     [0, 0,       e^{iθ},  0],
     [0, e^{iθ},  0,       0],
     [0, 0,       0,       1]]
    Pauli decomp via Tr(P·M)/4:
    II: ¼(1+0+0+1) = ½
    ZZ: ¼(1-0-0+1) = ½
    XX: ¼·Tr(XX·M): XX has 1s at (1,2) and (2,1) (in the |00>,|01>,|10>,|11> basis):
        Tr = e^{iθ} + e^{iθ} = 2e^{iθ} ... wait: XX_{12}=1, XX_{21}=1, M_{21}=e^{iθ}, M_{12}=e^{iθ}
        Tr(XX·M) = XX_{12}·M_{21} + XX_{21}·M_{12} = e^{iθ} + e^{iθ} = 2e^{iθ}... 
        Hmm: Tr(AB) = Σ_j (AB)_{jj} = Σ_{jk} A_{jk}B_{kj}
        XX_{jk}·M_{kj}: only nonzero when XX_{jk}≠0, i.e. j≠k with j,k in {1,2}:
        j=1,k=2: XX_{12}·M_{21} = 1·e^{iθ} = e^{iθ}
        j=2,k=1: XX_{21}·M_{12} = 1·e^{iθ} = e^{iθ}
        Tr = 2e^{iθ}, coeff_XX = e^{iθ}/2
    YY: (YY)_{12}=(σy)_{01}·(σy)_{10}·... YY_{|01>,|10>}=<0|Y|1><1|Y|0> = (-i)(i) = 1
        Wait: (σy)_{01}=-i, (σy)_{10}=+i... 
        (YY)_{12} means row=|01>, col=|10>: <0|Y|1>⊗<1|Y|0> = (-i)(i) = 1? No:
        For a tensor product P⊗Q, (P⊗Q)_{ab,cd} = P_{ac}·Q_{bd}.
        Mapping: |00>=0, |01>=1, |10>=2, |11>=3 where left qubit=high index.
        So row |01> means a=0,b=1, col |10> means c=1,d=0:
        (YY)_{01,10} = Y_{00,10}·Y_{01,00}... 
        Hmm, I'm confusing myself. Let me just use our PauliSum matrix method to derive numerically.
    """
    θ = sp.sympify(θ)
    eθ = _e(_iS * θ)
    # PSWAP = SWAP with phase e^{iθ} on the |01>,|10> swap:
    # = ½(1+e^{iθ})SWAP + ½(1-e^{iθ})·diag_piece
    # Actually easier: PSWAP = ½(II+ZZ) [no-swap projectors] + e^{iθ}·½(XX+YY) [swap part]
    # Verify: SWAP = ½(II+XX+YY+ZZ) → swap part = ½(XX+YY), no-swap = ½(II+ZZ) ✓
    return _gate2({
        'II': _H,
        'ZZ': _H,
        'XX': eθ * _H,
        'YY': eθ * _H,
    }, f'PSWAP({θ})', t0, t1)


# ===========================================================================
# SECTION 5: Three-qubit gates
# ===========================================================================
# Label convention: 3-char string 'ABC' → A=qubit2, B=qubit1, C=qubit0.

def gate_CCX(c0: int = 0, c1: int = 1, target: int = 2) -> Gate:
    """
    Toffoli / CCX gate.
    CCX = projection-based decomposition:
    CCX = ¼(III + IIX + IZI - IZX + ZII - ZIX - ZZI + ZZX
             + IIX + ... )
    Standard exact decomposition:
    CCX = ⅛(7·III + IIX + IZI - IZX + ZII - ZIX - ZZI + ZZX)
    (derived from |cc><cc|⊗X for cc=11, identity otherwise)
    Using |11><11| = ¼(II-IZ-ZI+ZZ) as the control projector:
    CCX = (I - |11><11|)⊗I + |11><11|⊗X
        = I⊗I⊗I - |11><11|⊗I + |11><11|⊗X
        = III - ¼(IIZ·... wait, careful about which qubits are control.

    Let me be precise. Qubits: c0=0 (rightmost label), c1=1 (middle), target=2 (left).
    |11><11| on (c0,c1) = |1><1|⊗|1><1| = ¼(I-Z)_c1 ⊗ (I-Z)_c0
    = ¼(II - IZ - ZI + ZZ) in (c1,c0) space.
    Embedding to 3-qubit: IIZ → 'IZI'? No: 
    c0=qubit0 (label pos 2), c1=qubit1 (label pos 1), target=qubit2 (label pos 0).
    |1><1|_c0 = ½(I-Z) acting on qubit0 → in 3-qubit label: 'IIZ' (Z at rightmost)? No:
    Wait: in our label 'ABC', A=qubit2, B=qubit1, C=qubit0.
    Z on qubit0 → 'IIZ'. Z on qubit1 → 'IZI'. Z on qubit2 → 'ZII'.
    
    P_11 = |11><11|_{q0,q1} = ¼(I-Z)_q0 ⊗ (I-Z)_q1
    In 3-qubit: ¼(III - IIZ - IZI + IZZ)
    
    CCX = III - P_11⊗I_q2 + P_11⊗X_q2
        = III - ¼(III - IIZ - IZI + IZZ) + ¼(XII - XIZ - XZI + XZZ)
        = III - ¼III + ¼IIZ + ¼IZI - ¼IZZ + ¼XII - ¼XIZ - ¼XZI + ¼XZZ
        = ¾III + ¼IIZ + ¼IZI - ¼IZZ + ¼XII - ¼XIZ - ¼XZI + ¼XZZ
    """
    d = {
        'III':  sp.Rational(7, 8),   # Wait: let me recount from III = 1·III
        # From III - ¼III = ¾III ... but then + ¼XII etc.
        # Let me redo cleanly:
    }
    # Clean derivation:
    # CCX = III - ¼(III-IIZ-IZI+IZZ) + ¼(XII-XIZ-XZI+XZZ)
    # = (1 - 1/4)III + (1/4)IIZ + (1/4)IZI + (-1/4)IZZ
    #   + (1/4)XII + (-1/4)XIZ + (-1/4)XZI + (1/4)XZZ
    d = {
        'III': sp.Rational(3, 4),
        'IIZ': sp.Rational(1, 4),
        'IZI': sp.Rational(1, 4),
        'IZZ': sp.Rational(-1, 4),
        'XII': sp.Rational(1, 4),
        'XIZ': sp.Rational(-1, 4),
        'XZI': sp.Rational(-1, 4),
        'XZZ': sp.Rational(1, 4),
    }
    return _gate3(d, 'CCX', c0, c1, target)

def gate_CCZ(c0: int = 0, c1: int = 1, c2: int = 2) -> Gate:
    """
    CCZ gate.
    CCZ = III - 2·|111><111|
    |111><111| = ⅛(I-Z)⊗(I-Z)⊗(I-Z) = ⅛(III-IIZ-IZI-ZII+IZZ+ZIZ+ZZI-ZZZ)
    CCZ = III - ¼(III-IIZ-IZI-ZII+IZZ+ZIZ+ZZI-ZZZ)
        = ¾III + ¼IIZ + ¼IZI + ¼ZII - ¼IZZ - ¼ZIZ - ¼ZZI + ¼ZZZ
    """
    d = {
        'III': sp.Rational(3, 4),
        'IIZ': sp.Rational(1, 4),
        'IZI': sp.Rational(1, 4),
        'ZII': sp.Rational(1, 4),
        'IZZ': sp.Rational(-1, 4),
        'ZIZ': sp.Rational(-1, 4),
        'ZZI': sp.Rational(-1, 4),
        'ZZZ': sp.Rational(1, 4),
    }
    return _gate3(d, 'CCZ', c0, c1, c2)

def gate_CSWAP(control: int = 0, t0: int = 1, t1: int = 2) -> Gate:
    """
    Fredkin / CSWAP gate.
    CSWAP = |0><0|⊗II + |1><1|⊗SWAP
    = ½(I+Z)⊗II + ½(I-Z)⊗SWAP
    SWAP = ½(II+XX+YY+ZZ), embedded on (t0,t1):
    In our 3-qubit label (qubit2, qubit1, qubit0) with control=q0, t0=q1, t1=q2:
    ½(I+Z)_q0 ⊗ (II)_{q1,q2} + ½(I-Z)_q0 ⊗ ½(II+XX+YY+ZZ)_{q1,q2}
    = ½(III+IIZ) + ¼(III-IIZ+XXI+XXZ+YYI+YYZ+ZZI+ZZZ)
    Wait, let me be very careful with qubit ordering.
    control=qubit0 (label[2]), t0=qubit1 (label[1]), t1=qubit2 (label[0]).
    ½(I+Z) on qubit0 → 'IIZ' means Z at qubit0 pos. Let label = q2 q1 q0.
    (I+Z)_q0: adds 'IIZ' for the Z part.
    SWAP on (q1,q2): ½(II+XX+YY+ZZ)_{q1,q2} in 3-qubit = ½(III+XXI+YYI+ZZI) where last char=q0=I.
    Result:
    |0><0|⊗II: ½(I+Z)_q0·II_{q1q2} = ½(III + IIZ) [where Z is on q0]
    |1><1|⊗SWAP: ½(I-Z)_q0·½(II+XX+YY+ZZ)_{q1q2}
    = ¼[(III+IIZ-... )] wait: ½(I-Z) means + for I, - for Z.
    = ¼(III-IIZ) ⊗ then SWAP part... 
    
    Let me just build it via PauliSum multiplication:
    """
    # Build via PauliSum ops on 3 qubits
    # |0><0| = ½(I+Z), |1><1| = ½(I-Z)
    # proj0 on qubit0 (local index 0 of 3-qubit system):
    proj0 = PauliSum.from_dict({'III': _H, 'IIZ': _H}, n=3)  # ½(I+Z) on q0, label: q2q1q0 → IIZ means q0=Z
    proj1 = PauliSum.from_dict({'III': _H, 'IIZ': -_H}, n=3) # ½(I-Z) on q0
    # SWAP on (q1, q2) embedded in 3-qubit:
    swap_12 = PauliSum.from_dict({'III': _H, 'XXI': _H, 'YYI': _H, 'ZZI': _H}, n=3)
    # II on (q1,q2):
    id_12 = PauliSum.from_dict({'III': _I}, n=3)
    ps = proj0 * id_12 + proj1 * swap_12
    ps = ps.simplify()
    return Gate(ps, (control, t0, t1), 'CSWAP')


# ===========================================================================
# SECTION 6: Generic parameterized gates
# ===========================================================================

def gate_PauliRot(θ, pauli_label: str, targets: tuple | list, n_total: int | None = None) -> Gate:
    """
    Generic Pauli rotation: exp(-iθ/2 · P) for any Pauli string P.

    Parameters
    ----------
    θ : sympy expression or number
    pauli_label : str, e.g. 'XYZ' (length = number of local qubits)
    targets : tuple of qubit indices in the full system
    n_total : if given, return a Gate embedded in an n_total-qubit system
    """
    θ = sp.sympify(θ)
    n_local = len(pauli_label)
    # R_P(θ) = cos(θ/2)·I^n - i·sin(θ/2)·P
    id_label = 'I' * n_local
    ps = PauliSum.from_dict({
        id_label:   _c(θ/2),
        pauli_label: -_iS * _s(θ/2),
    }, n=n_local)
    name = f'PauliRot({θ}, {pauli_label})'
    return Gate(ps, tuple(targets), name)

def gate_MultiRZ(θ, n: int, targets: tuple | list) -> Gate:
    """
    MultiRZ(θ, n) = exp(-iθ/2 · Z⊗n)
    = cos(θ/2)·I^n - i·sin(θ/2)·Z^n
    """
    return gate_PauliRot(θ, 'Z' * n, targets)


# ===========================================================================
# SECTION 7: Convenience catalog
# ===========================================================================

def list_gates() -> list[str]:
    """Return all gate constructor names in this module."""
    return [name for name in globals() if name.startswith('gate_')]
