"""
Tests for sympauli.pauli_sum
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.pauli_sum import PauliSum, embed_sum, _X2, _Y2, _Z2
from sympauli.pauli_string import PauliString


def test_all():

    θ = sp.Symbol('θ', real=True)
    φ = sp.Symbol('φ', real=True)

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

    def check_matrix(name, got, expected, tol=1e-12):
        nonlocal passed, failed
        if np.allclose(got, expected, atol=tol):
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: max diff = {np.max(np.abs(got - expected))}")
            failed += 1

    def check_sympy(name, expr, tol=1e-12):
        """Check that a SymPy expression simplifies to zero."""
        nonlocal passed, failed
        simplified = sp.simplify(expr)
        try:
            val = abs(complex(simplified.evalf()))
            ok = (simplified == 0) or val < tol
        except Exception:
            ok = (simplified == 0)
        if ok:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: simplifies to {simplified}")
            failed += 1

    print("=== PauliSum Tests ===\n")

    # --- Construction ---
    print("Construction:")
    X = PauliSum.from_pauli(PauliString.from_string('X'))
    Z = PauliSum.from_pauli(PauliString.from_string('Z'))
    I1 = PauliSum.identity(1)
    check("X len=1", len(X), 1)
    check("I len=1", len(I1), 1)

    # from_dict
    Rz = PauliSum.from_dict({'I': sp.cos(θ/2), 'Z': -sp.I * sp.sin(θ/2)}, n=1)
    check("Rz len=2", len(Rz), 2)

    print("\nAddition:")
    S = X + Z
    check("X+Z len=2", len(S), 2)
    S2 = X + X
    check("X+X len=1", len(S2.simplify()), 1)
    # coefficient should be 2
    coeff_X = list(S2._terms.values())[0]
    check("X+X coeff=2", coeff_X, sp.Integer(2))

    print("\nScalar multiplication:")
    S3 = sp.Rational(1, 2) * X
    coeff = list(S3._terms.values())[0]
    check("(1/2)X coeff", coeff, sp.Rational(1, 2))
    S4 = X * sp.Integer(3)
    coeff4 = list(S4._terms.values())[0]
    check("X*3 coeff", coeff4, sp.Integer(3))

    print("\nOperator multiplication:")
    # X * X = I
    XX = X * X
    XX_s = XX.simplify()
    check("X·X = I, len=1", len(XX_s), 1)
    p, c = XX_s.terms()[0]
    check("X·X = I label", p.label(), 'I')
    check("X·X = I coeff", sp.simplify(c - 1), 0)

    # X * Z = -iY  (XZ = -iY)
    XZ = X * Z
    XZ_s = XZ.simplify()
    check("X·Z label=Y", XZ_s.terms()[0][0].label(), 'Y')
    check("X·Z coeff=-i", sp.simplify(XZ_s.terms()[0][1] + sp.I), 0)

    # [X, Z] = -2iY
    comm = X.commutator(Z).simplify()
    check("[X,Z] label=Y", comm.terms()[0][0].label(), 'Y')
    check("[X,Z] coeff=-2i", sp.simplify(comm.terms()[0][1] + 2*sp.I), 0)

    # {X, Z} = 0
    anti = X.anticommutator(Z).simplify()
    check("{X,Z}=0", len(anti._terms), 0)

    print("\nRz gate Pauli sum:")
    # Rz(θ) = cos(θ/2)I - i·sin(θ/2)Z
    # Rz† = cos(θ/2)I + i·sin(θ/2)Z
    Rz_dag = Rz.adjoint()
    # Rz† · Rz should equal I (up to trig simplification)
    RzdRz = (Rz_dag * Rz).simplify('trig')
    check("Rz†·Rz = I, len=1", len(RzdRz), 1)
    p, c = RzdRz.terms()[0]
    check("Rz†·Rz label=I", p.label(), 'I')
    check_sympy("Rz†·Rz coeff=1", c - 1)

    print("\nHeisenberg conjugation (sneak preview):")
    # Rz†(θ) · X · Rz(θ) should give cos(θ)X - sin(θ)Y  ... wait let me check sign conv.
    # Rz(θ) = exp(-iθZ/2) = cos(θ/2)I - i sin(θ/2)Z
    # Rz† X Rz:
    conj = (Rz_dag * X * Rz).simplify('trig')
    labels = {p.label(): sp.trigsimp(c) for p, c in conj.terms()}
    # Expected: cos(θ)·X + sin(θ)·Y   (standard result)
    check("Rz†·X·Rz has X term", 'X' in labels, True)
    check("Rz†·X·Rz has Y term", 'Y' in labels, True)
    check_sympy("Rz†·X·Rz X-coeff = cos(θ)", labels.get('X', 0) - sp.cos(θ))
    check_sympy("Rz†·X·Rz Y-coeff = -sin(θ)", labels.get('Y', 0) + sp.sin(θ))

    print("\nSubstitution:")
    Rz0 = Rz.subs(θ, 0).simplify()
    check("Rz(0)=I, len=1", len(Rz0), 1)
    check("Rz(0) label=I", Rz0.terms()[0][0].label(), 'I')

    RzPi = Rz.subs(θ, sp.pi).simplify()
    check("Rz(π) len=1", len(RzPi), 1)
    check("Rz(π) label=Z", RzPi.terms()[0][0].label(), 'Z')

    print("\nto_matrix:")
    # X matrix
    X_mat = X.to_matrix()
    check_matrix("X matrix", X_mat, _X2)

    # Z matrix
    Z_mat = Z.to_matrix()
    check_matrix("Z matrix", Z_mat, _Z2)

    # Rz(π/2) matrix — standard rotation matrix
    Rz_num = Rz.to_matrix({θ: np.pi/2})
    expected_Rz = np.array([[np.exp(-1j*np.pi/4), 0], [0, np.exp(1j*np.pi/4)]])
    check_matrix("Rz(π/2) matrix", Rz_num, expected_Rz)

    print("\nTwo-qubit PauliSum:")
    XX2 = PauliSum.from_dict({'XX': sp.Integer(1)}, n=2)
    ZZ2 = PauliSum.from_dict({'ZZ': sp.Integer(1)}, n=2)
    IZ  = PauliSum.from_dict({'IZ': sp.Integer(1)}, n=2)
    # XX·ZZ = (X⊗X)(Z⊗Z) = (XZ)⊗(XZ) = (-iY)⊗(-iY) = -YY
    XXZZ = (XX2 * ZZ2).simplify()
    check("XX·ZZ len=1", len(XXZZ), 1)
    p, c = XXZZ.terms()[0]
    check("XX·ZZ label=YY", p.label(), 'YY')
    check_sympy("XX·ZZ coeff=-1", c + 1)

    print("\nembed_sum:")
    # Embed single-qubit X onto qubit 1 of a 3-qubit system: IXI
    X1 = PauliSum.from_pauli(PauliString.from_string('X'))
    X1_emb = embed_sum(X1, [1], 3)
    check("embed X→q1 len=1", len(X1_emb), 1)
    check("embed X→q1 label=IXI", X1_emb.terms()[0][0].label(), 'IXI')

    print(f"\n{'='*35}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")

