"""
Tests for sympauli.pauli_string
"""
import pytest
import sympy as sp
import numpy as np
from sympauli.pauli_string import PauliString, pauli_product, _phase_str


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

    print("=== PauliString Tests ===\n")

    # --- Construction ---
    print("Construction:")
    X = PauliString.from_string('X')
    Y = PauliString.from_string('Y')
    Z = PauliString.from_string('Z')
    I = PauliString.from_string('I')
    check("X label", X.label(), 'X')
    check("Y label", Y.label(), 'Y')
    check("Z label", Z.label(), 'Z')
    check("I label", I.label(), 'I')
    check("X phase", X.phase, 0)

    # Multi-qubit from_string
    XZ = PauliString.from_string('XZ')   # qubit 1 = X, qubit 0 = Z
    check("XZ qubit0", XZ.pauli_at(0), 'Z')
    check("XZ qubit1", XZ.pauli_at(1), 'X')

    print("\nSingle-qubit multiplication:")
    # XX = I (phase 0)
    XX = X * X
    check("X·X = +1·I",  XX.label(), 'I')
    check("X·X phase=0", XX.phase, 0)

    # XY = iZ
    XY = X * Y
    check("X·Y = iZ label",  XY.label(), 'Z')
    check("X·Y = iZ phase",  XY.phase, 1)

    # YX = -iZ
    YX = Y * X
    check("Y·X = -iZ label", YX.label(), 'Z')
    check("Y·X = -iZ phase", YX.phase, 3)

    # YZ = iX
    YZ = Y * Z
    check("Y·Z = iX label",  YZ.label(), 'X')
    check("Y·Z = iX phase",  YZ.phase, 1)

    # ZX = iY
    ZX = Z * X
    check("Z·X = iY label",  ZX.label(), 'Y')
    check("Z·X = iY phase",  ZX.phase, 1)

    # ZY = -iX
    ZY = Z * Y
    check("Z·Y = -iX label", ZY.label(), 'X')
    check("Z·Y = -iX phase", ZY.phase, 3)

    # YY = I (phase 0)
    YY = Y * Y
    check("Y·Y = +I label", YY.label(), 'I')
    check("Y·Y phase=0",    YY.phase, 0)

    # ZZ = I
    ZZ = Z * Z
    check("Z·Z = +I label", ZZ.label(), 'I')
    check("Z·Z phase=0",    ZZ.phase, 0)

    print("\nMulti-qubit multiplication:")
    # (X⊗Z) · (Y⊗X) = (XY)⊗(ZX) = (iZ)⊗(iY) = i²·ZY = -ZY
    XZ_p = PauliString.from_string('XZ')  # qubit1=X, qubit0=Z
    YX_p = PauliString.from_string('YX')  # qubit1=Y, qubit0=X
    prod = XZ_p * YX_p
    check("XZ·YX label", prod.label(), 'ZY')
    check("XZ·YX phase", prod.phase, 2)   # -1

    print("\nCommutativity:")
    check("X,Z anti-commute", X.commutes_with(Z), False)
    check("X,X commute",      X.commutes_with(X), True)
    check("X,I commute",      X.commutes_with(I), True)
    XI = PauliString.from_string('XI')
    IX = PauliString.from_string('IX')
    check("XI,IX commute",    XI.commutes_with(IX), True)
    XZ2 = PauliString.from_string('XZ')
    ZX2 = PauliString.from_string('ZX')
    check("XZ,ZX commute (both flip sign)", XZ2.commutes_with(ZX2), True)

    print("\nWeight:")
    check("I weight=0",    I.weight, 0)
    check("X weight=1",    X.weight, 1)
    check("XZ weight=2",   PauliString.from_string('XZ').weight, 2)
    check("IXI weight=1",  PauliString.from_string('IXI').weight, 1)

    print("\nAdjoint / conjugate:")
    # X, Y, Z are all Hermitian: P† = P (phase negated, but X/Z have phase 0)
    check("X† = X", X.adjoint().label(), 'X')
    check("Y† = Y", Y.adjoint().label(), 'Y')
    # iX: phase=1, adjoint should be phase=3 (-i)
    iX = PauliString(X.x_bits, X.z_bits, 1, phase=1)
    check("(iX)† phase=-i", iX.adjoint().phase, 3)
    # Y* = -Y (conjugate flips sign of Y)
    check("Y* label",  Y.conjugate().label(), 'Y')
    check("Y* phase",  Y.conjugate().phase, 2)  # -1

    print("\nTensor product:")
    XtZ = X.tensor(Z)  # X on qubit 1, Z on qubit 0 → 'XZ'
    check("X⊗Z label", XtZ.label(), 'XZ')
    check("X⊗Z n=2",   XtZ.n, 2)

    print("\nEmbed:")
    X1 = PauliString.from_string('X')
    X_emb = X1.embed([2], 4)  # X on qubit 2 of a 4-qubit system
    check("embed X→qubit2, label", X_emb.label(), 'IXII')
    check("embed X→qubit2, n=4",   X_emb.n, 4)

    Y1 = PauliString.from_string('Y')
    Y_emb = Y1.embed([0], 3)
    check("embed Y→qubit0, label", Y_emb.label(), 'IIY')

    print("\npauli_product helper:")
    chain = pauli_product(X, Y, Z)   # X·Y·Z = iZ·Z = i·I
    check("X·Y·Z = iI label", chain.label(), 'I')
    check("X·Y·Z = iI phase", chain.phase, 1)

    print(f"\n{'='*30}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        pytest.fail(f"{failed} test(s) failed")

