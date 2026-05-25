"""
example.py — demonstration of the pauli_engine package.

Run with:  python -m pauli_engine.example
"""

import sympy as sp
import numpy as np

from sympauli import PauliSum, evolve, gradient, expectation_value
from sympauli.gates import (
    gate_Ry, gate_Rz, gate_Rx, gate_CNOT, gate_RZZ, gate_RXX,
    gate_H, gate_CRy, gate_PauliRot,
)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    # ------------------------------------------------------------------
    section("1. Single-qubit Heisenberg evolution")
    # ------------------------------------------------------------------
    theta = sp.Symbol('theta', real=True)

    # Evolve Z through Ry(theta): Ry†·Z·Ry = cos(theta)·Z - sin(theta)·X
    obs = PauliSum.from_dict({'Z': 1}, n=1)
    circuit = [gate_Ry(theta, target=0)]
    result = evolve(obs, circuit, n_qubits=1)

    print("\nRy†(θ)·Z·Ry(θ) =")
    for p, c in result.terms():
        print(f"  ({sp.trigsimp(c)}) · {p.label()}")

    # ------------------------------------------------------------------
    section("2. Symbolic gradient (parameter-shift compatible)")
    # ------------------------------------------------------------------
    grad = gradient(obs, circuit, n_qubits=1, param_symbol=theta)
    print("\nd/dθ [Ry†(θ)·Z·Ry(θ)] =")
    for p, c in grad.terms():
        print(f"  ({sp.trigsimp(c)}) · {p.label()}")

    # ------------------------------------------------------------------
    section("3. VQE-style expectation value <0|H(θ)|0>")
    # ------------------------------------------------------------------
    state_0 = np.array([1.0, 0.0], dtype=complex)
    print("\n<0|Ry†(θ)·Z·Ry(θ)|0> = cos(θ)")
    for t in [0, np.pi/4, np.pi/2, np.pi]:
        ev = expectation_value(result, state_0, {theta: t}).real
        print(f"  θ={t:.4f}:  {ev:+.6f}  (expected cos(θ)={np.cos(t):+.6f})")

    # ------------------------------------------------------------------
    section("4. Two-qubit: Ising Hamiltonian through an ansatz")
    # ------------------------------------------------------------------
    phi = sp.Symbol('phi', real=True)
    n = 2

    # Transverse-field Ising Hamiltonian
    H_ising = PauliSum.from_dict({'ZZ': 1, 'XI': 1, 'IX': 1}, n=n)
    print(f"\nH = {H_ising}")

    # Ansatz: Ry(theta)⊗Ry(phi), then CNOT
    circuit2 = [
        gate_Ry(theta, target=0),
        gate_Ry(phi,   target=1),
        gate_CNOT(control=0, target=1),
    ]

    H_evolved = evolve(H_ising, circuit2, n_qubits=n, verbose=True)
    print(f"\nEvolved H has {len(H_evolved)} Pauli terms.")
    print("Terms:")
    for p, c in H_evolved.terms():
        print(f"  ({sp.trigsimp(c)}) · {p.label()}")

    # Numerical check
    t0, p0 = 0.4, 0.9
    state_00 = np.array([1, 0, 0, 0], dtype=complex)
    ev2 = expectation_value(H_evolved, state_00, {theta: t0, phi: p0}).real
    print(f"\n<00|H(θ={t0},φ={p0})|00> = {ev2:.6f}")

    # ------------------------------------------------------------------
    section("5. Generic PauliRot: exp(-i·theta/2 · ZZZ)")
    # ------------------------------------------------------------------
    n3 = 3
    obs3 = PauliSum.from_dict({'ZZZ': 1}, n=n3)
    circuit3 = [gate_PauliRot(theta, 'ZZZ', [0, 1, 2])]
    result3 = evolve(obs3, circuit3, n_qubits=n3)
    print("\nPauliRot(θ,ZZZ)†·ZZZ·PauliRot(θ,ZZZ) =")
    for p, c in result3.terms():
        print(f"  ({sp.trigsimp(c)}) · {p.label()}")
    print("(ZZZ commutes with its own rotation → ZZZ unchanged, as expected)")


if __name__ == "__main__":
    main()
