"""
heisenberg.py
-------------
Layer 4 of the Symbolic Pauli Heisenberg Evolution Engine.

Core operation: evolve an observable H through a parameterized circuit U(θ)
in the Heisenberg picture:

    H(θ) = U†(θ) · H · U(θ)

This is computed gate-by-gate, right to left through the circuit:

    H → G₁† H G₁ → G₂†(G₁† H G₁)G₂ → ...

Each step is the core "conjugation" operation:
    H ← G† H G

where G is a Gate (PauliSum on local qubits, embedded into the full system).

----------------------------------------------------------------------
Key design choices
----------------------------------------------------------------------

1.  Lazy embedding: gate PauliSums are embedded into the n-qubit space
    on-the-fly during evolution, so the gate library stays qubit-agnostic.

2.  Incremental simplification: after each conjugation step the PauliSum
    is simplified with trigsimp.  This keeps term counts manageable and
    avoids exponential blow-up in intermediate expressions.  The user can
    also request lazy simplification (only at the end) for speed.

3.  Full symbolic: all coefficients remain SymPy expressions; parameters
    θ are free symbols until the user calls .subs().

4.  Validation: evolve_numeric() substitutes random parameter values and
    compares against a direct matrix computation, giving a tight test loop.

----------------------------------------------------------------------
Public API
----------------------------------------------------------------------

    evolve(observable, circuit, n_qubits,
           simplify='trig', verbose=False)  ->  PauliSum

    evolve_numeric(observable, circuit, n_qubits, param_values)  ->  np.ndarray

    conjugate_by_gate(H, gate, n_qubits)  ->  PauliSum

    commutator(A, B)   ->  PauliSum   (A·B - B·A, convenience wrapper)

    gradient(observable, circuit, n_qubits, param_symbol)  ->  PauliSum
        Symbolic derivative ∂H(θ)/∂θ_k via parameter-shift or direct
        SymPy differentiation of the evolved PauliSum coefficients.
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from typing import Sequence

from .pauli_string import PauliString
from .pauli_sum import PauliSum, embed_sum
from .gates import Gate


# ---------------------------------------------------------------------------
# Core: single-step conjugation
# ---------------------------------------------------------------------------

def conjugate_by_gate(
    H: PauliSum,
    gate: Gate,
    n_qubits: int,
    simplify: str | None = "trig",
) -> PauliSum:
    """
    Compute G† · H · G where G is a Gate embedded into an n_qubits system.

    Parameters
    ----------
    H        : PauliSum on n_qubits qubits (the current observable)
    gate     : Gate  (local PauliSum + target qubit list)
    n_qubits : total number of qubits in the system
    simplify : 'trig' | 'full' | 'expand' | None
               Simplification applied after conjugation.

    Returns
    -------
    PauliSum  G† H G, on n_qubits qubits.
    """
    G  = embed_sum(gate.pauli_sum, list(gate.targets), n_qubits)
    Gd = G.adjoint()                         # Hermitian adjoint: conjugate all coefficients
    result = Gd * H * G
    if simplify:
        result = result.simplify(simplify)
    return result


# ---------------------------------------------------------------------------
# Core: full circuit evolution
# ---------------------------------------------------------------------------

def evolve(
    observable: PauliSum,
    circuit: Sequence[Gate],
    n_qubits: int,
    simplify: str | None = "trig",
    verbose: bool = False,
) -> PauliSum:
    """
    Evolve an observable H through a circuit in the Heisenberg picture:

        H(θ) = U†(θ) · H · U(θ),   U = G_n · G_{n-1} · ... · G_1

    Gates are applied right-to-left: G_1 first (innermost), G_n last.

    Parameters
    ----------
    observable : PauliSum  —  the initial observable H (on n_qubits qubits)
    circuit    : list of Gate  —  ordered gate sequence [G_1, G_2, ..., G_n]
    n_qubits   : int  —  total number of qubits
    simplify   : simplification strategy after each step ('trig', 'full',
                 'expand', or None for no intermediate simplification)
    verbose    : if True, print term count after each gate

    Returns
    -------
    PauliSum  — H evolved through the full circuit, with symbolic coefficients
    """
    H = observable

    if verbose:
        print(f"Initial observable: {len(H)} term(s)")

    for i, gate in enumerate(reversed(circuit)):
        H = conjugate_by_gate(H, gate, n_qubits, simplify=simplify)
        if verbose:
            print(f"  After gate {len(circuit)-i}: {gate.name:25s}  →  {len(H)} term(s)")

    return H


# ---------------------------------------------------------------------------
# Numeric reference: matrix-based evolution for validation
# ---------------------------------------------------------------------------

def evolve_numeric(
    observable: PauliSum,
    circuit: Sequence[Gate],
    n_qubits: int,
    param_values: dict,
) -> np.ndarray:
    """
    Compute U†(θ₀) · H · U(θ₀) numerically as a 2ⁿ × 2ⁿ matrix.

    Parameters
    ----------
    observable   : PauliSum  — the observable H
    circuit      : list of Gate
    n_qubits     : int
    param_values : dict {sympy.Symbol: float}  — numerical parameter values

    Returns
    -------
    np.ndarray, shape (2**n, 2**n)
    """
    dim = 2 ** n_qubits
    H_mat = observable.to_matrix(param_values)

    # Build U = G_n · ... · G_1 as a matrix product
    U = np.eye(dim, dtype=complex)
    for gate in circuit:
        G_emb = embed_sum(gate.pauli_sum, list(gate.targets), n_qubits)
        G_mat = G_emb.to_matrix(param_values)
        U = G_mat @ U

    return U.conj().T @ H_mat @ U


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate(
    symbolic_result: PauliSum,
    observable: PauliSum,
    circuit: Sequence[Gate],
    n_qubits: int,
    param_values: dict,
    tol: float = 1e-8,
) -> tuple[bool, float]:
    """
    Check that the symbolic result matches the numeric evolution.

    Returns (passed: bool, max_error: float).
    """
    sym_mat  = symbolic_result.to_matrix(param_values)
    num_mat  = evolve_numeric(observable, circuit, n_qubits, param_values)
    err = float(np.max(np.abs(sym_mat - num_mat)))
    return err < tol, err


# ---------------------------------------------------------------------------
# Symbolic gradient: ∂H(θ)/∂θ_k
# ---------------------------------------------------------------------------

def gradient(
    observable: PauliSum,
    circuit: Sequence[Gate],
    n_qubits: int,
    param_symbol: sp.Symbol,
    simplify: str | None = "trig",
    verbose: bool = False,
) -> PauliSum:
    """
    Compute the symbolic gradient of the evolved observable with respect to
    a single parameter symbol:

        ∂/∂θ_k [U†(θ) H U(θ)]

    This is done by symbolically differentiating each coefficient of the
    evolved PauliSum with respect to param_symbol.

    Parameters
    ----------
    observable   : PauliSum  — the initial observable
    circuit      : list of Gate
    n_qubits     : int
    param_symbol : sympy.Symbol  — the parameter to differentiate with respect to
    simplify     : simplification after differentiation

    Returns
    -------
    PauliSum  — the gradient as a PauliSum with symbolic coefficients
    """
    evolved = evolve(observable, circuit, n_qubits, simplify=simplify, verbose=verbose)

    result = PauliSum(n_qubits)
    for (xb, zb, n), coeff in evolved._terms.items():
        dcoeff = sp.diff(coeff, param_symbol)
        if simplify:
            fn = {"trig": sp.trigsimp, "full": sp.simplify, "expand": sp.expand}.get(simplify, sp.trigsimp)
            dcoeff = fn(dcoeff)
        if dcoeff != 0:
            result._terms[(xb, zb, n)] = dcoeff

    return result.prune()


# ---------------------------------------------------------------------------
# Expectation value: <ψ|H(θ)|ψ> for a given state
# ---------------------------------------------------------------------------

def expectation_value(
    evolved_H: PauliSum,
    state: np.ndarray,
    param_values: dict | None = None,
) -> complex:
    """
    Compute <ψ|H(θ)|ψ> numerically given the evolved observable and a state vector.

    Parameters
    ----------
    evolved_H    : PauliSum  — result of evolve()
    state        : np.ndarray, shape (2**n,)  — normalized state vector
    param_values : optional dict for substituting symbols

    Returns
    -------
    complex  — the expectation value
    """
    H_mat = evolved_H.to_matrix(param_values or {})
    return complex(state.conj() @ H_mat @ state)


# ---------------------------------------------------------------------------
# Utility: commutator
# ---------------------------------------------------------------------------

def commutator(A: PauliSum, B: PauliSum) -> PauliSum:
    """[A, B] = A·B - B·A"""
    return A.commutator(B)
