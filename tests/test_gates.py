"""
Tests for sympauli.gates
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.gates import *
from sympauli.pauli_sum import PauliSum


def test_all():


    passed = 0
    failed = 0

    def check(name, got, expected):
        nonlocal passed, failed
        if got == expected:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: got {got!r}, expected {expected!r}")
            failed += 1

    def check_matrix(name, got, expected, tol=1e-10):
        nonlocal passed, failed
        if np.allclose(got, expected, atol=tol):
            print(f"  ✓ {name}")
            passed += 1
        else:
            diff = np.max(np.abs(got - expected))
            print(f"  ✗ {name}: max diff = {diff:.2e}")
            print(f"    got:\n{np.round(got,4)}")
            print(f"    expected:\n{np.round(expected,4)}")
            failed += 1

    def mat(gate, subs=None):
        return gate.pauli_sum.to_matrix(subs or {})

    # Reference matrices
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    H2 = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
    S2 = np.array([[1,0],[0,1j]], dtype=complex)
    T2 = np.array([[1,0],[0,np.exp(1j*np.pi/4)]], dtype=complex)
    SX2 = np.array([[1+1j, 1-1j],[1-1j, 1+1j]], dtype=complex) / 2

    def Rx_ref(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*X
    def Ry_ref(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Y
    def Rz_ref(t): return np.cos(t/2)*I2 - 1j*np.sin(t/2)*Z

    CNOT_ref = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    SWAP_ref  = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    CZ_ref    = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)

    t0 = 1.2345   # generic angle

    print("=== Gate Library Tests ===\n")

    # -----------------------------------------------------------------------
    print("Single-qubit Clifford gates:")
    check_matrix("I",    mat(gate_I()),    I2)
    check_matrix("X",    mat(gate_X()),    X)
    check_matrix("Y",    mat(gate_Y()),    Y)
    check_matrix("Z",    mat(gate_Z()),    Z)
    check_matrix("H",    mat(gate_H()),    H2)
    check_matrix("S",    mat(gate_S()),    S2)
    check_matrix("Sdg",  mat(gate_Sdg()), S2.conj().T)
    check_matrix("T",    mat(gate_T()),    T2)
    check_matrix("Tdg",  mat(gate_Tdg()), T2.conj().T)
    check_matrix("SX",   mat(gate_SX()),  SX2)
    check_matrix("SXdg", mat(gate_SXdg()),SX2.conj().T)

    # -----------------------------------------------------------------------
    print("\nSingle-qubit parameterized gates:")
    for ang in [0.0, np.pi/4, np.pi/2, np.pi, 1.2345]:
        check_matrix(f"Rx({ang:.3f})", mat(gate_Rx(ang)), Rx_ref(ang))
        check_matrix(f"Ry({ang:.3f})", mat(gate_Ry(ang)), Ry_ref(ang))
        check_matrix(f"Rz({ang:.3f})", mat(gate_Rz(ang)), Rz_ref(ang))

    # PhaseShift
    P_ref = lambda l: np.array([[1,0],[0,np.exp(1j*l)]], dtype=complex)
    for lam in [0.0, np.pi/3, np.pi]:
        check_matrix(f"PhaseShift({lam:.3f})", mat(gate_PhaseShift(lam)), P_ref(lam))

    # U3 special cases — compare up to global phase
    def up_to_global_phase(name, got, exp):
        # Find a non-tiny element and compute the phase ratio
        mask = np.abs(exp) > 1e-6
        if not np.any(mask): return check_matrix(name, got, exp)
        ratio = (got[mask] / exp[mask])
        phase = ratio[0]
        check_matrix(name, got, np.exp(1j*np.angle(phase)) * exp, tol=1e-7)
    up_to_global_phase("U3=X (θ=π,φ=0,λ=π)", mat(gate_U3(np.pi, 0, np.pi)), X)
    up_to_global_phase("U3=H (θ=π/2,φ=0,λ=π)", mat(gate_U3(np.pi/2, 0, np.pi)), H2)

    # -----------------------------------------------------------------------
    print("\nTwo-qubit Clifford gates:")
    check_matrix("CNOT", mat(gate_CNOT()), CNOT_ref)
    check_matrix("CZ",   mat(gate_CZ()),   CZ_ref)
    check_matrix("SWAP", mat(gate_SWAP()), SWAP_ref)

    CY_ref = np.array([[1,0,0,0],[0,0,0,-1j],[0,0,1,0],[0,1j,0,0]], dtype=complex)
    check_matrix("CY", mat(gate_CY()), CY_ref)

    iSWAP_ref = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
    check_matrix("iSWAP", mat(gate_iSWAP()), iSWAP_ref)

    # -----------------------------------------------------------------------
    print("\nTwo-qubit parameterized gates:")

    # CRx/CRy/CRz with ctrl=q0: applies rotation to q1 when q0=1 (odd indices 1,3)
    def CRx_ref(t):
        M = np.eye(4, dtype=complex)
        c, s = np.cos(t/2), np.sin(t/2)
        M[1,1] = c; M[1,3] = -1j*s; M[3,1] = -1j*s; M[3,3] = c
        return M
    def CRy_ref(t):
        M = np.eye(4, dtype=complex)
        c, s = np.cos(t/2), np.sin(t/2)
        M[1,1] = c; M[1,3] = -s; M[3,1] = s; M[3,3] = c
        return M
    def CRz_ref(t):
        M = np.eye(4, dtype=complex)
        M[1,1] = np.exp(-1j*t/2); M[3,3] = np.exp(1j*t/2)
        return M

    for ang in [0.0, np.pi/3, np.pi/2, np.pi]:
        check_matrix(f"CRx({ang:.3f})", mat(gate_CRx(ang)), CRx_ref(ang))
        check_matrix(f"CRy({ang:.3f})", mat(gate_CRy(ang)), CRy_ref(ang))
        check_matrix(f"CRz({ang:.3f})", mat(gate_CRz(ang)), CRz_ref(ang))

    # RXX, RYY, RZZ
    def RXX_ref(t): return np.cos(t/2)*np.eye(4,dtype=complex) - 1j*np.sin(t/2)*np.kron(X,X)
    def RYY_ref(t): return np.cos(t/2)*np.eye(4,dtype=complex) - 1j*np.sin(t/2)*np.kron(Y,Y)
    def RZZ_ref(t): return np.cos(t/2)*np.eye(4,dtype=complex) - 1j*np.sin(t/2)*np.kron(Z,Z)

    for ang in [0.0, np.pi/4, np.pi/2, t0]:
        check_matrix(f"RXX({ang:.3f})", mat(gate_RXX(ang)), RXX_ref(ang))
        check_matrix(f"RYY({ang:.3f})", mat(gate_RYY(ang)), RYY_ref(ang))
        check_matrix(f"RZZ({ang:.3f})", mat(gate_RZZ(ang)), RZZ_ref(ang))

    # RZX: exp(-iθ/2 ZX) — Z on q1 (left/high), X on q0 (right/low)
    # In our convention qubit0 is rightmost, so kron(Z,X):
    def RZX_ref(t): return np.cos(t/2)*np.eye(4,dtype=complex) - 1j*np.sin(t/2)*np.kron(X,Z)
    check_matrix(f"RZX(π/2)", mat(gate_RZX(np.pi/2)), RZX_ref(np.pi/2))

    # IsingXY
    def IsingXY_ref(t):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([
            [1,    0,    0,   0],
            [0,    c,  1j*s,  0],
            [0,  1j*s,  c,   0],
            [0,    0,    0,   1]], dtype=complex)
    for ang in [0.0, np.pi/4, np.pi/2, np.pi]:
        check_matrix(f"IsingXY({ang:.3f})", mat(gate_IsingXY(ang)), IsingXY_ref(ang))

    # PSWAP
    def PSWAP_ref(t):
        e = np.exp(1j*t)
        return np.array([
            [1, 0, 0, 0],
            [0, 0, e, 0],
            [0, e, 0, 0],
            [0, 0, 0, 1]], dtype=complex)
    for ang in [0.0, np.pi/4, np.pi/2]:
        check_matrix(f"PSWAP({ang:.3f})", mat(gate_PSWAP(ang)), PSWAP_ref(ang))

    # XXMinusYY
    def XXMinusYY_ref(t, b):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([
            [c,               0, 0, -1j*s*np.exp(-1j*b)],
            [0,               1, 0,  0                  ],
            [0,               0, 1,  0                  ],
            [-1j*s*np.exp(1j*b), 0, 0,  c               ]], dtype=complex)
    for ang, bet in [(0.0,0.0),(np.pi/3, np.pi/4),(np.pi/2, np.pi/3)]:
        check_matrix(f"XXMinusYY({ang:.2f},{bet:.2f})",
                     mat(gate_XXMinusYY(ang, bet)), XXMinusYY_ref(ang, bet))

    # XXPlusYY
    def XXPlusYY_ref(t, b):
        c, s = np.cos(t/2), np.sin(t/2)
        return np.array([
            [1, 0,               0,               0],
            [0, c,  -1j*s*np.exp(-1j*b),          0],
            [0, -1j*s*np.exp(1j*b), c,             0],
            [0, 0,               0,               1]], dtype=complex)
    for ang, bet in [(0.0,0.0),(np.pi/3, np.pi/4),(np.pi/2, np.pi/3)]:
        check_matrix(f"XXPlusYY({ang:.2f},{bet:.2f})",
                     mat(gate_XXPlusYY(ang, bet)), XXPlusYY_ref(ang, bet))

    # -----------------------------------------------------------------------
    print("\nThree-qubit gates:")

    CCX_ref = np.eye(8, dtype=complex)
    CCX_ref[3,3]=0; CCX_ref[7,7]=0; CCX_ref[3,7]=1; CCX_ref[7,3]=1
    check_matrix("CCX/Toffoli", mat(gate_CCX()), CCX_ref)

    CCZ_ref = np.eye(8, dtype=complex)
    CCZ_ref[7,7] = -1
    check_matrix("CCZ", mat(gate_CCZ()), CCZ_ref)

    CSWAP_ref = np.eye(8, dtype=complex)
    # ctrl=q0, swaps q1,q2: |011>(3) <-> |101>(5) when q0=1
    CSWAP_ref[3,3]=0; CSWAP_ref[5,5]=0; CSWAP_ref[3,5]=1; CSWAP_ref[5,3]=1
    check_matrix("CSWAP/Fredkin", mat(gate_CSWAP()), CSWAP_ref)

    # -----------------------------------------------------------------------
    print("\nGeneric PauliRot:")
    pr = gate_PauliRot(t0, 'ZZ', [0, 1])
    check_matrix(f"PauliRot(ZZ)", mat(pr), RZZ_ref(t0))

    pr3 = gate_PauliRot(t0, 'ZZZ', [0,1,2])
    ZZZ = np.kron(Z, np.kron(Z, Z))
    ZZZ_ref = np.cos(t0/2)*np.eye(8, dtype=complex) - 1j*np.sin(t0/2)*ZZZ
    check_matrix(f"PauliRot(ZZZ)", mat(pr3), ZZZ_ref)

    # -----------------------------------------------------------------------
    print("\nUnitarity check (U†U = I) for all 1- and 2-qubit gates:")
    gates_to_check = [
        gate_H(), gate_S(), gate_Sdg(), gate_T(), gate_Tdg(), gate_SX(), gate_SXdg(),
        gate_Rx(t0), gate_Ry(t0), gate_Rz(t0), gate_PhaseShift(t0),
        gate_CNOT(), gate_CY(), gate_CZ(), gate_SWAP(), gate_iSWAP(),
        gate_CRx(t0), gate_CRy(t0), gate_CRz(t0),
        gate_RXX(t0), gate_RYY(t0), gate_RZZ(t0),
        gate_IsingXY(t0), gate_PSWAP(t0),
    ]
    for g in gates_to_check:
        M = mat(g)
        I_ref = np.eye(M.shape[0], dtype=complex)
        check_matrix(f"  unitary: {g.name}", M.conj().T @ M, I_ref)

    # -----------------------------------------------------------------------
    print("\nEmbed test:")
    # Embed Rz on qubit 2 of a 4-qubit system
    rz_emb = gate_Rz(t0, target=0).embed(4)  # target=0 locally → embed to global qubit 2
    rz_emb2 = gate_Rz(t0, target=2)
    # embed to qubit 2 of 4:
    rz_g = gate_Rz(t0, target=0)
    ps_emb = embed_sum(rz_g.pauli_sum, [2], 4)
    M_emb = ps_emb.to_matrix()
    I4 = np.eye(4, dtype=complex)
    M_expected = np.kron(I2, np.kron(Rz_ref(t0), I4))  # Rz on qubit2 of 4: I⊗Rz⊗I⊗I
    check_matrix("Embed Rz→qubit2 of 4", M_emb, M_expected)

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")


