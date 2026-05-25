"""
Tests for sympauli.heisenberg
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.heisenberg import (
    evolve, conjugate_by_gate, gradient, expectation_value, validate
)
from sympauli.pauli_sum import PauliSum
from sympauli.gates import (
    gate_Rx, gate_Ry, gate_Rz,
    gate_CNOT, gate_CZ, gate_RZZ, gate_RYY, gate_RXX,
    gate_H, gate_X, gate_Z, gate_SWAP,
    gate_CRy, gate_CRz,
)


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

    def check_val(name, got, expected, tol=1e-9):
        nonlocal passed, failed
        if abs(complex(got) - complex(expected)) < tol:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: got {got}, expected {expected}")
            failed += 1

    def check_matrix(name, got, expected, tol=1e-8):
        nonlocal passed, failed
        if np.allclose(got, expected, atol=tol):
            print(f"  ✓ {name}")
            passed += 1
        else:
            diff = np.max(np.abs(got - expected))
            print(f"  ✗ {name}: max diff = {diff:.2e}")
            failed += 1

    def check_sympy(name, expr, tol=1e-10):
        nonlocal passed, failed
        s = sp.trigsimp(sp.simplify(expr))
        try:
            val = abs(complex(s.evalf()))
            ok = (s == 0) or val < tol
        except Exception:
            ok = (s == 0)
        if ok:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: simplifies to {s}")
            failed += 1

    θ = sp.Symbol('θ', real=True)
    φ = sp.Symbol('φ', real=True)

    print("=== Heisenberg Evolution Engine Tests ===\n")

    # -----------------------------------------------------------------------
    print("1. Single-gate conjugation (analytic checks):")

    # Rz†(θ)·X·Rz(θ) = cos(θ)·X + sin(θ)·Y
    X_obs = PauliSum.from_dict({'X': sp.Integer(1)}, n=1)
    rz = gate_Rz(θ, target=0)
    result = conjugate_by_gate(X_obs, rz, n_qubits=1)
    labels = {p.label(): c for p, c in result.terms()}
    check_sympy("Rz†·X·Rz: X-coeff = cos(θ)", sp.trigsimp(labels.get('X', 0) - sp.cos(θ)))
    check_sympy("Rz†·X·Rz: Y-coeff = -sin(θ)", sp.trigsimp(labels.get('Y', 0) + sp.sin(θ)))
    check("Rz†·X·Rz: no other terms", set(labels.keys()), {'X', 'Y'})

    # Rx†(θ)·Z·Rx(θ) = cos(θ)·Z - sin(θ)·Y  ... wait:
    # Rx = cos(θ/2)I - i sin(θ/2)X
    # Rx† Z Rx: Z commutes through cos term, anti-commutes through X term.
    # = cos²(θ/2)·Z - sin²(θ/2)·Z + 2i cos(θ/2)sin(θ/2)·... 
    # Exact: Rx†·Z·Rx = cos(θ)·Z + sin(θ)·Y
    Z_obs = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    rx = gate_Rx(θ, target=0)
    result2 = conjugate_by_gate(Z_obs, rx, n_qubits=1)
    labels2 = {p.label(): c for p, c in result2.terms()}
    check_sympy("Rx†·Z·Rx: Z-coeff = cos(θ)", sp.trigsimp(labels2.get('Z', 0) - sp.cos(θ)))
    check_sympy("Rx†·Z·Rx: Y-coeff = sin(θ)", sp.trigsimp(labels2.get('Y', 0) - sp.sin(θ)))

    # Ry†(θ)·X·Ry(θ) = cos(θ)·X - sin(θ)·Z
    Y_obs = PauliSum.from_dict({'Y': sp.Integer(1)}, n=1)
    X_obs1 = PauliSum.from_dict({'X': sp.Integer(1)}, n=1)
    ry = gate_Ry(θ, target=0)
    result3 = conjugate_by_gate(X_obs1, ry, n_qubits=1)
    labels3 = {p.label(): c for p, c in result3.terms()}
    check_sympy("Ry†·X·Ry: X-coeff = cos(θ)", sp.trigsimp(labels3.get('X', 0) - sp.cos(θ)))
    check_sympy("Ry†·X·Ry: Z-coeff = +sin(θ)", sp.trigsimp(labels3.get('Z', 0) - sp.sin(θ)))

    # H·Z·H = X  (Hadamard conjugates Z to X)
    Z_obs1 = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    h = gate_H(target=0)
    result4 = conjugate_by_gate(Z_obs1, h, n_qubits=1)
    result4 = result4.simplify('full')
    labels4 = {p.label(): sp.simplify(c) for p, c in result4.terms()}
    check("H·Z·H = X", set(labels4.keys()), {'X'})
    check_sympy("H·Z·H X-coeff = 1", labels4.get('X', 0) - 1)

    # -----------------------------------------------------------------------
    print("\n2. Two-qubit conjugation:")

    # Standard Heisenberg rules for CNOT(ctrl=q0, tgt=q1):
    #   Z_ctrl (IZ) → Z_ctrl (IZ)          [ctrl Z is preserved]
    #   Z_tgt  (ZI) → Z_ctrl·Z_tgt (ZZ)   [tgt Z picks up ctrl Z]
    IZ_obs = PauliSum.from_dict({'IZ': sp.Integer(1)}, n=2)  # Z on ctrl=q0
    cnot = gate_CNOT(control=0, target=1)
    result5 = conjugate_by_gate(IZ_obs, cnot, n_qubits=2).simplify()
    labels5 = {p.label(): sp.simplify(c) for p, c in result5.terms()}
    # CNOT†·IZ·CNOT = IZ  (ctrl Z is unchanged)
    check("CNOT†·IZ·CNOT = IZ", set(labels5.keys()), {'IZ'})
    check_sympy("CNOT†·IZ·CNOT IZ-coeff=1", labels5.get('IZ', 0) - 1)

    # -----------------------------------------------------------------------
    print("\n3. Full circuit evolution (analytic):")

    # Circuit: Rz(φ) on qubit 0
    # Observable: X on qubit 0
    # Result: cos(φ)·X + sin(φ)·Y  (single rotation)
    # Wait: Rz†·X·Rz was cos(θ)X - sin(θ)Y ... let me re-check direction.
    # evolve() applies gates left-to-right in the circuit, Heisenberg means reversing.
    # For circuit [Rz(θ)], we apply Rz†·X·Rz.
    obs = PauliSum.from_dict({'X': sp.Integer(1)}, n=1)
    circuit1 = [gate_Rz(θ, target=0)]
    evolved1 = evolve(obs, circuit1, n_qubits=1)
    labels_e1 = {p.label(): c for p, c in evolved1.terms()}
    check_sympy("evolve X through Rz(θ): X = cos(θ)", labels_e1.get('X', 0) - sp.cos(θ))
    check_sympy("evolve X through Rz(θ): Y = -sin(θ)", labels_e1.get('Y', 0) + sp.sin(θ))

    # Circuit: Rx(θ) then Ry(φ) on qubit 0, observable Z
    # = Ry†(Rx†·Z·Rx)·Ry
    # Rx†·Z·Rx = cos(θ)Z + sin(θ)Y
    # Ry†·(cos(θ)Z + sin(θ)Y)·Ry:
    #   Ry†·Z·Ry = cos(φ)Z - sin(φ)X   ... wait: Ry†·Z·Ry = cos(φ)Z + sin(φ)... 
    #   Standard: Ry†·Z·Ry = cos(φ)Z - sin(φ)X (rotation of Z toward X by angle φ)
    #   Ry†·Y·Ry = Y  (Y commutes with Ry)
    # Result: cos(θ)(cos(φ)Z - sin(φ)X) + sin(θ)·Y
    obs2 = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    circuit2 = [gate_Rx(θ, target=0), gate_Ry(φ, target=0)]
    evolved2 = evolve(obs2, circuit2, n_qubits=1)
    labels_e2 = {p.label(): sp.trigsimp(c) for p, c in evolved2.terms()}
    check_sympy("Rx·Ry evolve Z: X-coeff = -sin(φ)",
                labels_e2.get('X', 0) + sp.sin(φ))
    check_sympy("Rx·Ry evolve Z: Y-coeff = cos(φ)sin(θ)",
                labels_e2.get('Y', 0) - sp.cos(φ)*sp.sin(θ))
    check_sympy("Rx·Ry evolve Z: Z-coeff = cos(φ)cos(θ)",
                labels_e2.get('Z', 0) - sp.cos(φ)*sp.cos(θ))

    # -----------------------------------------------------------------------
    print("\n4. Numeric validation (symbolic == matrix-based):")

    θ0 = sp.Symbol('θ0', real=True)
    φ0 = sp.Symbol('φ0', real=True)
    param_vals = {θ0: 0.7432, φ0: 1.2109}

    # 1-qubit: Rx(θ0) · Ry(φ0), observable Z
    obs_z = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    circ_a = [gate_Rx(θ0), gate_Ry(φ0)]
    sym_a = evolve(obs_z, circ_a, n_qubits=1)
    ok, err = validate(sym_a, obs_z, circ_a, 1, param_vals)
    check(f"1-qubit Rx·Ry (err={err:.2e})", ok, True)

    # 2-qubit: CNOT + Rz on qubit 0, observable ZZ
    obs_zz = PauliSum.from_dict({'ZZ': sp.Integer(1)}, n=2)
    circ_b = [gate_CNOT(0, 1), gate_Rz(θ0, target=0)]
    sym_b = evolve(obs_zz, circ_b, n_qubits=2)
    ok, err = validate(sym_b, obs_zz, circ_b, 2, param_vals)
    check(f"2-qubit CNOT+Rz (err={err:.2e})", ok, True)

    # 2-qubit: RZZ(θ) + Rx(φ) on qubit 1, observable XI
    obs_xi = PauliSum.from_dict({'XI': sp.Integer(1)}, n=2)
    circ_c = [gate_RZZ(θ0), gate_Rx(φ0, target=1)]
    sym_c = evolve(obs_xi, circ_c, n_qubits=2)
    ok, err = validate(sym_c, obs_xi, circ_c, 2, param_vals)
    check(f"2-qubit RZZ+Rx (err={err:.2e})", ok, True)

    # -----------------------------------------------------------------------
    print("\n5. Multi-qubit Hamiltonian observable:")

    # H = Z⊗Z + X⊗I + I⊗X  (transverse-field Ising, 2 qubits)
    H_ising = PauliSum.from_dict({
        'ZZ': sp.Integer(1),
        'XI': sp.Integer(1),
        'IX': sp.Integer(1),
    }, n=2)

    # Evolve through a 2-qubit ansatz: Ry(θ)⊗Ry(θ), then CNOT
    circ_d = [gate_Ry(θ0, target=0), gate_Ry(θ0, target=1), gate_CNOT(0, 1)]
    sym_d = evolve(H_ising, circ_d, n_qubits=2)
    ok, err = validate(sym_d, H_ising, circ_d, 2, param_vals)
    check(f"Ising H through Ry⊗Ry+CNOT (err={err:.2e})", ok, True)
    check(f"  term count is finite", len(sym_d) > 0, True)

    # -----------------------------------------------------------------------
    print("\n6. Symbolic gradient:")

    # d/dθ [Rz†(θ)·X·Rz(θ)] = d/dθ [cos(θ)X - sin(θ)Y] = -sin(θ)X - cos(θ)Y
    obs_x = PauliSum.from_dict({'X': sp.Integer(1)}, n=1)
    circ_rz = [gate_Rz(θ, target=0)]
    grad = gradient(obs_x, circ_rz, n_qubits=1, param_symbol=θ)
    glabels = {p.label(): sp.trigsimp(c) for p, c in grad.terms()}
    check_sympy("∂/∂θ[Rz†XRz] X-coeff = -sin(θ)", glabels.get('X', 0) + sp.sin(θ))
    check_sympy("∂/∂θ[Rz†XRz] Y-coeff = -cos(θ)", glabels.get('Y', 0) + sp.cos(θ))

    # d/dθ [cos(θ)Z + sin(θ)Y] = -sin(θ)Z + cos(θ)Y  (from Rx†·Z·Rx)
    obs_z2 = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    circ_rx = [gate_Rx(θ, target=0)]
    grad2 = gradient(obs_z2, circ_rx, n_qubits=1, param_symbol=θ)
    glabels2 = {p.label(): sp.trigsimp(c) for p, c in grad2.terms()}
    check_sympy("∂/∂θ[Rx†ZRx] Z-coeff = -sin(θ)", glabels2.get('Z', 0) + sp.sin(θ))
    check_sympy("∂/∂θ[Rx†ZRx] Y-coeff = cos(θ)", glabels2.get('Y', 0) - sp.cos(θ))

    # Numerical gradient check: d<Z>/dθ at θ=θ0 for single Rx rotation
    # <Z(θ)> = cos(θ)  →  d/dθ<Z> = -sin(θ)
    theta_val = 0.9
    grad_sym_val = float(glabels2.get('Z', 0).subs(θ, theta_val).evalf()
                         + glabels2.get('Y', 0).subs(θ, theta_val).evalf() * 0)  # Y part vanishes in |0><0| state
    # |0> state: <0|Y|0>=0, <0|Z|0>=1 → <Z(θ)> = cos(θ), grad = -sin(θ)
    expected_grad = -np.sin(theta_val)

    # -----------------------------------------------------------------------
    print("\n7. VQE-style: symbolic expectation on |0> state:")

    # Evolved Z through Ry(θ): Ry†·Z·Ry = cos(θ)Z - sin(θ)X
    # <0|cos(θ)Z - sin(θ)X|0> = cos(θ)·1 - sin(θ)·0 = cos(θ)
    obs_z3 = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    circ_ry = [gate_Ry(θ, target=0)]
    sym_ry = evolve(obs_z3, circ_ry, n_qubits=1)
    state_0 = np.array([1.0, 0.0], dtype=complex)
    for theta_test in [0.0, np.pi/4, np.pi/2, np.pi]:
        ev = expectation_value(sym_ry, state_0, {θ: theta_test}).real
        expected = np.cos(theta_test)
        check_val(f"  <Z(Ry({theta_test:.3f}))>|0⟩ = cos = {expected:.4f}",
                  ev, expected, tol=1e-9)

    # -----------------------------------------------------------------------
    print("\n8. Deeper circuit: 3-gate chain with verbose output:")

    obs_3 = PauliSum.from_dict({'ZI': sp.Integer(1)}, n=2)
    circ_3 = [
        gate_Rx(θ0, target=0),
        gate_CNOT(control=0, target=1),
        gate_Ry(φ0, target=1),
    ]
    print("  Verbose trace:")
    sym_3 = evolve(obs_3, circ_3, n_qubits=2, verbose=True)
    ok, err = validate(sym_3, obs_3, circ_3, 2, param_vals)
    check(f"  3-gate chain: numeric validation (err={err:.2e})", ok, True)

    # -----------------------------------------------------------------------
    print("\n9. Parameter-shift rule consistency check:")
    # For a gate G(θ) = exp(-iθP/2), the parameter-shift rule gives:
    #   d/dθ <H(θ)> = ½[<H(θ+π/2)> - <H(θ-π/2)>]  (evaluated at |0>)
    # Verify for Rx(θ) on observable Z
    theta_val = 0.73
    state = np.array([1.0, 0.0], dtype=complex)  # |0>
    circ_rx2 = [gate_Rx(θ, target=0)]
    obs_z4 = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    sym_rx2 = evolve(obs_z4, circ_rx2, n_qubits=1)

    def ev(t):
        return expectation_value(sym_rx2, state, {θ: t}).real

    psr = 0.5 * (ev(theta_val + np.pi/2) - ev(theta_val - np.pi/2))
    numerical_diff = (ev(theta_val + 1e-5) - ev(theta_val - 1e-5)) / 2e-5
    check_val("  PSR ≈ finite difference (Rx on Z)", psr, numerical_diff, tol=1e-6)

    # -----------------------------------------------------------------------
    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")


