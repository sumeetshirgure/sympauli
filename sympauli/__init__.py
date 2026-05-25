"""
sympauli — Symbolic Pauli Heisenberg Evolution Engine
======================================================

Evolves quantum observables through parameterized circuits in the Heisenberg
picture, keeping all coefficients as exact SymPy expressions.

Layers
------
1. pauli_string  : PauliString  — n-qubit Pauli strings (symplectic bitmask)
2. pauli_sum     : PauliSum     — symbolic linear combinations over SymPy
3. gates         : Gate library — 40+ standard gates (Qiskit + PennyLane coverage)
4. heisenberg    : Engine       — Heisenberg evolution, gradients, expectation values

Quick start
-----------
    from sympauli import PauliSum, evolve
    from sympauli.gates import gate_Ry, gate_CNOT
    import sympy as sp

    theta = sp.Symbol('theta', real=True)
    n = 2

    H = PauliSum.from_dict({'ZZ': 1, 'XI': 1, 'IX': 1}, n=n)
    circuit = [gate_Ry(theta, target=0), gate_Ry(theta, target=1), gate_CNOT(0, 1)]
    H_evolved = evolve(H, circuit, n_qubits=n)
    print(H_evolved)
"""

from .pauli_string import PauliString, pauli_product
from .pauli_sum import PauliSum, embed_sum
from .heisenberg import (
    evolve, conjugate_by_gate, gradient,
    expectation_value, validate, evolve_numeric,
)
from . import gates

__version__ = "0.1.0"

__all__ = [
    "PauliString", "pauli_product",
    "PauliSum", "embed_sum",
    "evolve", "conjugate_by_gate", "gradient",
    "expectation_value", "validate", "evolve_numeric",
    "gates",
]
