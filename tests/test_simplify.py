"""
Tests for sympauli.simplify
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.simplify import (
    as_rotation, conjugate_by_gate_fast, is_clifford_gate, simplify_coeffs,
)
from sympauli.heisenberg import conjugate_by_gate
from sympauli.pauli_string import PauliString
from sympauli.pauli_sum import PauliSum
from sympauli.gates import (
    gate_Rx, gate_Ry, gate_Rz,
    gate_RXX, gate_RYY, gate_RZZ, gate_RZX,
    gate_PauliRot, gate_MultiRZ,
    gate_CNOT, gate_CZ, gate_H, gate_S, gate_T, gate_SX, gate_SWAP,
    gate_PhaseShift, gate_CRz,
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

    print("=== Layer 5 Simplifier Tests ===\n")

    # -----------------------------------------------------------------------
    print("1. Rotation detection (as_rotation):")

    rot = as_rotation(gate_Rz(θ, target=0))
    check("as_rotation(Rz(θ)) is not None", rot is not None, True)
    if rot is not None:
        Q, angle = rot
        check("as_rotation(Rz(θ)) generator = Z", Q.label(), 'Z')
        check("as_rotation(Rz(θ)) generator phase-free", Q.phase, 0)
        check_sympy("as_rotation(Rz(θ)) angle = θ", angle - θ)

    rot_xx = as_rotation(gate_RXX(θ))
    check("as_rotation(RXX(θ)) generator = XX",
          rot_xx[0].label() if rot_xx else None, 'XX')
    rot_zx = as_rotation(gate_RZX(θ))
    check("as_rotation(RZX(θ)) generator = XZ",
          rot_zx[0].label() if rot_zx else None, 'XZ')
    rot_pr = as_rotation(gate_PauliRot(θ, 'XYZ', [0, 1, 2]))
    check("as_rotation(PauliRot(θ,XYZ)) generator = XYZ",
          rot_pr[0].label() if rot_pr else None, 'XYZ')
    rot_mrz = as_rotation(gate_MultiRZ(θ, 3, [0, 1, 2]))
    check("as_rotation(MultiRZ(θ,3)) generator = ZZZ",
          rot_mrz[0].label() if rot_mrz else None, 'ZZZ')

    # A numeric angle has already collapsed cos(θ/2) to a Float, so detection
    # goes through the acos branch.
    rot_num = as_rotation(gate_Rx(0.3, target=0))
    check("as_rotation(Rx(0.3)) generator = X",
          rot_num[0].label() if rot_num else None, 'X')
    if rot_num is not None:
        check_val("as_rotation(Rx(0.3)) angle = 0.3", rot_num[1], 0.3, tol=1e-9)
    rot_neg = as_rotation(gate_Ry(-1.1, target=0))
    check("as_rotation(Ry(-1.1)) generator = Y",
          rot_neg[0].label() if rot_neg else None, 'Y')
    if rot_neg is not None:
        check_val("as_rotation(Ry(-1.1)) angle = -1.1", rot_neg[1], -1.1, tol=1e-9)

    # Non-rotations, including the two-term gates that merely look like one.
    check("as_rotation(CNOT) is None", as_rotation(gate_CNOT()) is None, True)
    check("as_rotation(H) is None", as_rotation(gate_H()) is None, True)
    check("as_rotation(S) is None", as_rotation(gate_S()) is None, True)
    check("as_rotation(SX) is None", as_rotation(gate_SX()) is None, True)
    check("as_rotation(PhaseShift(λ)) is None",
          as_rotation(gate_PhaseShift(φ)) is None, True)
    check("as_rotation(CRz(θ)) is None", as_rotation(gate_CRz(θ)) is None, True)
    # An angle of unknown reality is rejected: the exact path's adjoint() would
    # leave conjugate(w) behind, so the closed form would not agree with it.
    w = sp.Symbol('w')
    check("as_rotation(Rz(w)) is None for non-real w",
          as_rotation(gate_Rz(w)) is None, True)

    # -----------------------------------------------------------------------
    print("\n2. Commuting terms are copied unchanged:")

    Z_obs = PauliSum.from_dict({'Z': sp.Integer(1)}, n=1)
    kept = conjugate_by_gate_fast(Z_obs, gate_Rz(θ, target=0), 1)
    check("Rz†·Z·Rz has one term", len(kept), 1)
    labels_k = {p.label(): c for p, c in kept.terms()}
    check("Rz†·Z·Rz = Z", set(labels_k.keys()), {'Z'})
    check_sympy("Rz†·Z·Rz Z-coeff = 1 (θ absent)", labels_k.get('Z', 0) - 1)
    check("Rz†·Z·Rz coefficient is θ-free", labels_k.get('Z', 0).free_symbols, set())

    # The identity always commutes, whatever the generator.
    I_obs = PauliSum.from_dict({'II': sp.Integer(1)}, n=2)
    kept_i = conjugate_by_gate_fast(I_obs, gate_RXX(θ), 2)
    check("RXX†·II·RXX = II", {p.label() for p, _ in kept_i.terms()}, {'II'})

    # ZZ commutes with the RZZ generator; XI does not touch a 3-qubit RZZ pair.
    ZZ_obs = PauliSum.from_dict({'ZZ': sp.Integer(1)}, n=2)
    kept_zz = conjugate_by_gate_fast(ZZ_obs, gate_RZZ(θ), 2)
    check("RZZ†·ZZ·RZZ = ZZ", {p.label() for p, _ in kept_zz.terms()}, {'ZZ'})

    # -----------------------------------------------------------------------
    print("\n3. Anticommuting terms take the closed form:")

    # Rz†(θ)·X·Rz(θ) = cos(θ)·X − sin(θ)·Y
    X_obs = PauliSum.from_dict({'X': sp.Integer(1)}, n=1)
    turned = conjugate_by_gate_fast(X_obs, gate_Rz(θ, target=0), 1)
    labels_t = {p.label(): c for p, c in turned.terms()}
    check("Rz†·X·Rz spans {X, Y}", set(labels_t.keys()), {'X', 'Y'})
    check_sympy("Rz†·X·Rz X-coeff = cos(θ)", labels_t.get('X', 0) - sp.cos(θ))
    check_sympy("Rz†·X·Rz Y-coeff = -sin(θ)", labels_t.get('Y', 0) + sp.sin(θ))

    # Rx†(θ)·Z·Rx(θ) = cos(θ)·Z + sin(θ)·Y   (matches test_heisenberg)
    turned2 = conjugate_by_gate_fast(Z_obs, gate_Rx(θ, target=0), 1)
    labels_t2 = {p.label(): c for p, c in turned2.terms()}
    check_sympy("Rx†·Z·Rx Z-coeff = cos(θ)", labels_t2.get('Z', 0) - sp.cos(θ))
    check_sympy("Rx†·Z·Rx Y-coeff = sin(θ)", labels_t2.get('Y', 0) - sp.sin(θ))

    # Two-qubit generator: RZZ†·XI·RZZ = cos(θ)·XI − sin(θ)·YZ
    # (XI · ZZ = (X·Z)⊗(I·Z) = (-iY)⊗Z, and −i·sin(θ)·(−i) = −sin(θ).)
    XI_obs = PauliSum.from_dict({'XI': sp.Integer(1)}, n=2)
    turned3 = conjugate_by_gate_fast(XI_obs, gate_RZZ(θ), 2)
    labels_t3 = {p.label(): c for p, c in turned3.terms()}
    check("RZZ†·XI·RZZ spans {XI, YZ}", set(labels_t3.keys()), {'XI', 'YZ'})
    check_sympy("RZZ†·XI·RZZ XI-coeff = cos(θ)", labels_t3.get('XI', 0) - sp.cos(θ))
    check_sympy("RZZ†·XI·RZZ YZ-coeff = -sin(θ)", labels_t3.get('YZ', 0) + sp.sin(θ))

    # -----------------------------------------------------------------------
    print("\n4. Equivalence to the exact path (fast − exact ≡ 0):")

    battery_1q = [
        ("Rx(θ)", gate_Rx(θ, target=0)),
        ("Ry(θ)", gate_Ry(θ, target=0)),
        ("Rz(θ)", gate_Rz(θ, target=0)),
    ]
    obs_1q = PauliSum.from_dict({
        'X': sp.Integer(2),
        'Y': sp.cos(φ),
        'Z': sp.Rational(1, 3),
        'I': sp.Integer(1),
    }, n=1)
    for name, g in battery_1q:
        fast = conjugate_by_gate_fast(obs_1q, g, 1)
        exact = conjugate_by_gate(obs_1q, g, 1)
        check(f"  {name}: fast == exact", len((fast - exact).simplify('full')), 0)

    battery_2q = [
        ("RXX(θ)", gate_RXX(θ, 0, 1)),
        ("RYY(θ)", gate_RYY(θ, 0, 1)),
        ("RZZ(θ)", gate_RZZ(θ, 0, 1)),
        ("RZZ(θ) swapped targets", gate_RZZ(θ, 1, 0)),
        ("Rx(θ) on qubit 1", gate_Rx(θ, target=1)),
    ]
    obs_2q = PauliSum.from_dict({
        'XI': sp.Integer(1),
        'IZ': sp.Integer(-2),
        'YY': sp.sin(φ),
        'ZX': sp.Rational(1, 2),
    }, n=2)
    for name, g in battery_2q:
        fast = conjugate_by_gate_fast(obs_2q, g, 2)
        exact = conjugate_by_gate(obs_2q, g, 2)
        check(f"  {name}: fast == exact", len((fast - exact).simplify('full')), 0)

    obs_3q = PauliSum.from_dict({
        'XYZ': sp.Integer(1),
        'ZZI': sp.Integer(2),
        'IIX': sp.Rational(1, 2),
        'YIY': sp.cos(φ),
    }, n=3)
    battery_3q = [
        ("PauliRot(θ,XYZ)", gate_PauliRot(θ, 'XYZ', [0, 1, 2])),
        ("MultiRZ(θ,3)", gate_MultiRZ(θ, 3, [0, 1, 2])),
        ("RYY(θ) on (1,2)", gate_RYY(θ, 1, 2)),
    ]
    for name, g in battery_3q:
        fast = conjugate_by_gate_fast(obs_3q, g, 3)
        exact = conjugate_by_gate(obs_3q, g, 3)
        check(f"  {name}: fast == exact", len((fast - exact).simplify('full')), 0)

    # Non-rotations must fall through to the exact path, byte for byte.
    for name, g in [("CNOT", gate_CNOT(0, 1)), ("H on qubit 0", gate_H(0)),
                    ("S on qubit 1", gate_S(1))]:
        fast = conjugate_by_gate_fast(obs_2q, g, 2)
        exact = conjugate_by_gate(obs_2q, g, 2)
        check(f"  {name}: fast delegates to exact", len((fast - exact).simplify('full')), 0)

    # Numeric-angle rotations go down the fast path too; check against a matrix.
    obs_num = PauliSum.from_dict({'X': sp.Integer(1), 'Z': sp.Integer(1)}, n=1)
    g_num = gate_Rx(0.37, target=0)
    fast_num = conjugate_by_gate_fast(obs_num, g_num, 1)
    G = g_num.pauli_sum.to_matrix({})
    check_matrix("Rx(0.37) fast == G†·H·G matrix",
                 fast_num.to_matrix({}),
                 G.conj().T @ obs_num.to_matrix({}) @ G)

    # -----------------------------------------------------------------------
    print("\n5. Clifford detection:")

    check("is_clifford_gate(CNOT)", is_clifford_gate(gate_CNOT()), True)
    check("is_clifford_gate(H)", is_clifford_gate(gate_H()), True)
    check("is_clifford_gate(CZ)", is_clifford_gate(gate_CZ()), True)
    check("is_clifford_gate(S)", is_clifford_gate(gate_S()), True)
    check("is_clifford_gate(SWAP)", is_clifford_gate(gate_SWAP()), True)
    check("is_clifford_gate(SX)", is_clifford_gate(gate_SX()), True)
    check("is_clifford_gate(Rx(0.3)) is False (generic angle)",
          is_clifford_gate(gate_Rx(0.3)), False)
    check("is_clifford_gate(T) is False", is_clifford_gate(gate_T()), False)
    check("is_clifford_gate(Rz(π/4)) is False", is_clifford_gate(gate_Rz(sp.pi/4)), False)
    check("is_clifford_gate(Rx(π/2)) is True",
          is_clifford_gate(gate_Rx(sp.pi/2)), True)
    check("is_clifford_gate(Rx(θ)) is False (symbolic)",
          is_clifford_gate(gate_Rx(θ)), False)

    # -----------------------------------------------------------------------
    print("\n6. simplify_coeffs:")

    # sin²+cos²−1 is zero but only 'full' (or _is_zero's own fallback) sees it.
    trivial = PauliSum.from_dict({
        'X': sp.sin(θ)**2 + sp.cos(θ)**2 - 1,
        'Z': sp.Integer(3),
    }, n=1)
    reduced = simplify_coeffs(trivial, method='full')
    check("sin²+cos²−1 term is dropped", {p.label() for p, _ in reduced.terms()}, {'Z'})
    check_sympy("surviving Z-coeff = 3", dict(
        (p.label(), c) for p, c in reduced.terms()).get('Z', 0) - 3)

    # 'trig' collapses the same coefficient; 'none' still prunes it, because
    # pruning goes through _is_zero rather than through the simplifier.
    check("method='trig' also drops it", len(simplify_coeffs(trivial, method='trig')), 1)
    check("method='none' prunes without rewriting",
          len(simplify_coeffs(trivial, method='none')), 1)

    # 'none' leaves a surviving coefficient syntactically untouched.
    unruly = PauliSum.from_dict({'X': (θ + 1)**2}, n=1)
    check("method='none' does not expand",
          simplify_coeffs(unruly, method='none').terms()[0][1], (θ + 1)**2)
    check("method='expand' expands",
          simplify_coeffs(unruly, method='expand').terms()[0][1],
          sp.expand((θ + 1)**2))

    # 'cancel' clears a rational function of the parameter.
    rational_ps = PauliSum.from_dict({'Y': (θ**2 - 1) / (θ - 1)}, n=1)
    check_sympy("method='cancel' gives θ+1",
                simplify_coeffs(rational_ps, method='cancel').terms()[0][1] - (θ + 1))

    # rational=True folds a float back to an exact rational.
    floaty = PauliSum.from_dict({'Z': sp.Float(0.5)}, n=1)
    check("rational=True gives Rational(1,2)",
          simplify_coeffs(floaty, method='none', rational=True).terms()[0][1],
          sp.Rational(1, 2))
    check("rational=False keeps the Float",
          simplify_coeffs(floaty, method='none').terms()[0][1], sp.Float(0.5))
    # nsimplify on symbolic input must not raise out of simplify_coeffs.
    check("rational=True survives symbolic input",
          len(simplify_coeffs(PauliSum.from_dict({'X': sp.cos(θ)/3}, n=1),
                              method='trig', rational=True)), 1)

    # prune_tol is honoured for small numeric coefficients.
    tiny = PauliSum.from_dict({'X': sp.Float(1e-9), 'Z': sp.Integer(1)}, n=1)
    check("prune_tol=1e-6 drops the 1e-9 term",
          {p.label() for p, _ in simplify_coeffs(tiny, method='none',
                                                 prune_tol=1e-6).terms()}, {'Z'})
    check("default prune_tol keeps the 1e-9 term", len(simplify_coeffs(tiny, method='none')), 2)

    # No mutation of the input.
    check("simplify_coeffs leaves its input alone", len(trivial), 2)

    # -----------------------------------------------------------------------
    print("\n7. conjugate_by_gate_fast honours the extended vocabulary:")

    # 'cancel' and 'none' are simplify_coeffs methods that PauliSum.simplify
    # does not know; they must mean the same thing whichever path a gate takes.
    obs_pair = PauliSum.from_dict({'IZ': sp.Integer(1), 'XI': sp.Integer(1)}, n=2)
    for gname, g in [("RZZ(θ) (closed form)", gate_RZZ(θ)), ("CNOT (exact path)", gate_CNOT())]:
        none_res = conjugate_by_gate_fast(obs_pair, g, 2, simplify='none')
        trig_res = conjugate_by_gate_fast(obs_pair, g, 2, simplify='trig')
        check(f"  {gname}: 'none' agrees with 'trig' on term set",
              {p.label() for p, _ in none_res.terms()},
              {p.label() for p, _ in trig_res.terms()})
        check(f"  {gname}: 'cancel' agrees with 'trig' on term set",
              {p.label() for p, _ in
               conjugate_by_gate_fast(obs_pair, g, 2, simplify='cancel').terms()},
              {p.label() for p, _ in trig_res.terms()})
        raw = conjugate_by_gate_fast(obs_pair, g, 2, simplify=None)
        check(f"  {gname}: simplify=None skips the pass entirely",
              len(raw) >= len(trig_res), True)

    # The three shared methods must behave exactly as PauliSum.simplify did,
    # since simplify_coeffs is a superset and not a replacement.
    for method in ('trig', 'full', 'expand'):
        via_layer5 = conjugate_by_gate_fast(obs_pair, gate_CNOT(), 2, simplify=method)
        via_heisenberg = conjugate_by_gate(obs_pair, gate_CNOT(), 2, simplify=method)
        check(f"  simplify='{method}' matches conjugate_by_gate exactly",
              via_layer5._terms, via_heisenberg._terms)

    # -----------------------------------------------------------------------
    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")
