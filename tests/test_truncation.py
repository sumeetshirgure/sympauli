"""
Tests for sympauli.truncation
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.truncation import (
    truncate, truncate_weight, truncate_coeff, evolve_truncated,
)
from sympauli.heisenberg import evolve, evolve_numeric
from sympauli.pauli_sum import PauliSum
from sympauli.gates import (
    gate_Rx, gate_Ry, gate_Rz, gate_RZZ, gate_CNOT, gate_H,
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
    ψ = sp.Symbol('ψ', real=True)

    print("=== Truncation Tests ===\n")

    # -----------------------------------------------------------------------
    print("1. Weight truncation:")

    mixed = PauliSum.from_dict({
        'IIZ': sp.Integer(1),          # weight 1
        'IXY': sp.Integer(2),          # weight 2
        'XYZ': sp.Integer(3),          # weight 3
        'III': sp.Integer(4),          # weight 0
    }, n=3)
    check("hand-built sum has 4 terms", len(mixed), 4)

    kept2 = truncate_weight(mixed, 2)
    check("max_weight=2 keeps weight ≤ 2",
          {p.label() for p, _ in kept2.terms()}, {'III', 'IIZ', 'IXY'})
    check("max_weight=1 keeps weight ≤ 1",
          {p.label() for p, _ in truncate_weight(mixed, 1).terms()}, {'III', 'IIZ'})
    check("max_weight=0 keeps only the identity",
          {p.label() for p, _ in truncate_weight(mixed, 0).terms()}, {'III'})
    check("max_weight=3 keeps everything", len(truncate_weight(mixed, 3)), 4)
    check("coefficients survive untouched",
          dict((p.label(), c) for p, c in kept2.terms())['IXY'], sp.Integer(2))
    check("input is not mutated", len(mixed), 4)
    check("truncate_weight returns a new object", truncate_weight(mixed, 3) is mixed, False)

    # -----------------------------------------------------------------------
    print("\n2. Magnitude truncation, numeric coefficients:")

    numeric = PauliSum.from_dict({
        'IZ': sp.Float(0.5),
        'ZI': sp.Float(1e-6),
        'XX': sp.Float(0.3),
    }, n=2)
    dropped = truncate_coeff(numeric, 1e-3)
    check("min_magnitude=1e-3 drops only the 1e-6 term",
          {p.label() for p, _ in dropped.terms()}, {'IZ', 'XX'})
    check("min_magnitude=1.0 drops all three", len(truncate_coeff(numeric, 1.0)), 0)
    check("min_magnitude=1e-9 keeps all three", len(truncate_coeff(numeric, 1e-9)), 3)
    # A complex coefficient is judged on its modulus.
    complex_ps = PauliSum.from_dict({'X': 3 * sp.I / 1000, 'Z': sp.Integer(1)}, n=1)
    check("modulus of 0.003i is below 1e-2",
          {p.label() for p, _ in truncate_coeff(complex_ps, 1e-2).terms()}, {'Z'})
    check("input is not mutated", len(numeric), 3)

    # -----------------------------------------------------------------------
    print("\n3. Magnitude truncation, symbolic keep-rule:")

    symbolic = PauliSum.from_dict({'X': sp.sin(θ), 'Z': sp.Integer(1)}, n=1)
    check("sin(θ) with no subs is KEPT (cannot be judged)",
          {p.label() for p, _ in truncate_coeff(symbolic, 1e-3).terms()}, {'X', 'Z'})
    check("subs={θ: 0} makes sin(θ) judgeable and it is dropped",
          {p.label() for p, _ in truncate_coeff(symbolic, 1e-3, {θ: 0}).terms()}, {'Z'})
    check("subs={θ: 1.0} keeps it", len(truncate_coeff(symbolic, 1e-3, {θ: 1.0})), 2)
    # The survivor keeps its *symbolic* coefficient; subs is only a probe.
    survivor = dict((p.label(), c) for p, c in
                    truncate_coeff(symbolic, 1e-3, {θ: 1.0}).terms())
    check("surviving coefficient is still sin(θ), not its value",
          survivor['X'], sp.sin(θ))
    # A partial subs leaves free symbols behind → still unjudgeable → kept.
    two_sym = PauliSum.from_dict({'X': sp.sin(θ) * sp.sin(φ)}, n=1)
    check("partially substituted coefficient is kept",
          len(truncate_coeff(two_sym, 1e-3, {θ: 1.0})), 1)
    check("fully substituted coefficient is judged",
          len(truncate_coeff(two_sym, 1e-3, {θ: 1.0, φ: 0})), 0)

    # -----------------------------------------------------------------------
    print("\n4. Combined truncate():")

    combo = PauliSum.from_dict({
        'IIZ': sp.Float(0.9),          # weight 1, large   → kept
        'IXY': sp.Float(1e-8),         # weight 2, tiny    → dropped by magnitude
        'XYZ': sp.Float(0.9),          # weight 3, large   → dropped by weight
    }, n=3)
    both = truncate(combo, max_weight=2, min_magnitude=1e-3)
    check("weight and magnitude cutoffs both apply",
          {p.label() for p, _ in both.terms()}, {'IIZ'})
    check("max_weight only", len(truncate(combo, max_weight=2)), 2)
    check("min_magnitude only", len(truncate(combo, min_magnitude=1e-3)), 2)
    check("no cutoffs is a copy", len(truncate(combo)), 3)
    check("no cutoffs returns a new object", truncate(combo) is combo, False)

    # -----------------------------------------------------------------------
    print("\n5. evolve_truncated bounds the term count:")

    # Brickwork of Rx layers and RZZ pairs on n=4.  The bound is structural —
    # weight filtering is a key filter — so simplify=None is used here to keep
    # the sweep cheap; the surviving *labels* do not depend on that choice.
    n = 4
    obs_z = PauliSum.from_dict({'IIIZ': sp.Integer(1)}, n=n)

    def brickwork(depth):
        circuit = []
        for d in range(depth):
            for q in range(n):
                circuit.append(gate_Rx(θ, target=q))
            pairs = [(0, 1), (2, 3)] if d % 2 == 0 else [(1, 2), (0, 3)]
            for a, b in pairs:
                circuit.append(gate_RZZ(φ, a, b))
        return circuit

    BOUND = 12
    counts_trunc = []
    counts_exact = []
    for depth in (1, 2, 3, 4):
        circ = brickwork(depth)
        trunc = evolve_truncated(obs_z, circ, n, max_weight=2, simplify=None)
        full = evolve_truncated(obs_z, circ, n, simplify=None)
        counts_trunc.append(len(trunc))
        counts_exact.append(len(full))
        max_w = max((p.weight for p, _ in trunc.terms()), default=0)
        check(f"  depth {depth}: every surviving term has weight ≤ 2", max_w <= 2, True)
    print(f"  term counts: truncated {counts_trunc}  untruncated {counts_exact}")
    check(f"truncated count stays ≤ {BOUND} at every depth",
          all(c <= BOUND for c in counts_trunc), True)
    check(f"untruncated count grows past {BOUND}",
          counts_exact[-1] > BOUND, True)
    check("untruncated count grows monotonically",
          all(b >= a for a, b in zip(counts_exact, counts_exact[1:])), True)

    # min_magnitude prunes too, given values for the parameters.
    circ3 = brickwork(3)
    tiny_cut = evolve_truncated(obs_z, circ3, n, max_weight=2, min_magnitude=1e-2,
                                subs={θ: 0.05, φ: 0.05}, simplify=None)
    check("magnitude cutoff shrinks the depth-3 result further",
          len(tiny_cut) < counts_trunc[2], True)

    # -----------------------------------------------------------------------
    print("\n6. Truncation is lossless when the cutoff cannot bite:")

    # max_weight = n_qubits admits every Pauli string, so the result must equal
    # the exact evolution term for term.
    #
    # Each gate gets its own parameter.  With one symbol shared between two
    # rotations the two paths still agree, but they reach different (equal)
    # trigonometric forms and the difference lands on half-angle expressions
    # such as sin(θ)·cos(θ)/tan(θ/2) − 2·sin²(θ/2)·cos(θ) that sp.simplify
    # cannot reduce to 0, so a symbolic zero-test would report a false failure.
    # The shared-parameter case is covered numerically at the end of §7.
    obs_zz = PauliSum.from_dict({'ZZ': sp.Integer(1), 'XI': sp.Integer(1)}, n=2)
    circ_2q = [gate_Rx(θ, target=0), gate_RZZ(φ, 0, 1), gate_Ry(ψ, target=1)]
    exact = evolve(obs_zz, circ_2q, 2)
    loose = evolve_truncated(obs_zz, circ_2q, 2, max_weight=2)
    check("max_weight=2 on n=2: same term count", len(loose), len(exact))
    check("max_weight=2 on n=2: difference simplifies to zero",
          len((loose - exact).simplify('full')), 0)
    # And the same with a Clifford in the circuit, which takes the exact path.
    circ_mixed = [gate_H(0), gate_Rz(θ, target=0), gate_CNOT(0, 1)]
    exact_m = evolve(obs_zz, circ_mixed, 2)
    loose_m = evolve_truncated(obs_zz, circ_mixed, 2, max_weight=2)
    check("H + Rz + CNOT: difference simplifies to zero",
          len((loose_m - exact_m).simplify('full')), 0)

    # A cutoff that *does* bite is a genuine approximation, and it should show.
    tight = evolve_truncated(obs_zz, circ_2q, 2, max_weight=1)
    check("max_weight=1 loses terms", len(tight) < len(exact), True)

    # -----------------------------------------------------------------------
    print("\n7. Fast-path parity inside evolve_truncated:")

    fast = evolve_truncated(obs_zz, circ_2q, 2, max_weight=2, use_fast=True)
    slow = evolve_truncated(obs_zz, circ_2q, 2, max_weight=2, use_fast=False)
    check("rotation-only circuit: use_fast agrees with the exact path",
          len((fast - slow).simplify('full')), 0)

    fast_m = evolve_truncated(obs_zz, circ_mixed, 2, max_weight=2, use_fast=True)
    slow_m = evolve_truncated(obs_zz, circ_mixed, 2, max_weight=2, use_fast=False)
    check("mixed circuit: use_fast agrees with the exact path",
          len((fast_m - slow_m).simplify('full')), 0)

    # Numeric cross-check against the dense matrix product.
    param_vals = {θ: 0.4361, φ: 1.1027, ψ: 0.2508}
    check_matrix("truncated (loose) == numeric matrix evolution",
                 fast.to_matrix(param_vals),
                 evolve_numeric(obs_zz, circ_2q, 2, param_vals))

    # The shared-parameter circuit skipped above: sp.simplify cannot prove the
    # difference zero, but the matrices agree to machine precision.
    circ_shared = [gate_Rx(θ, target=0), gate_RZZ(φ, 0, 1), gate_Ry(θ, target=1)]
    shared_vals = {θ: 0.4361, φ: 1.1027}
    shared = evolve_truncated(obs_zz, circ_shared, 2, max_weight=2)
    check_matrix("shared parameter, loose cutoff == numeric matrix evolution",
                 shared.to_matrix(shared_vals),
                 evolve_numeric(obs_zz, circ_shared, 2, shared_vals))
    check_matrix("shared parameter, use_fast=False agrees numerically",
                 evolve_truncated(obs_zz, circ_shared, 2, max_weight=2,
                                  use_fast=False).to_matrix(shared_vals),
                 evolve_numeric(obs_zz, circ_shared, 2, shared_vals))

    # -----------------------------------------------------------------------
    print(f"\n{'='*45}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")
