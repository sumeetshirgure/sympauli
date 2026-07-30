# sympauli

**Symbolic Pauli Heisenberg Evolution Engine**

Evolves quantum observables through parameterized circuits in the Heisenberg picture,
keeping all coefficients as exact SymPy expressions.
Vibe coded using Claude AI in one afternoon.

## Install dependencies

```bash
pip install sympy numpy
```

## Usage

```python
from sympauli import PauliSum, evolve, gradient, expectation_value
from sympauli.gates import gate_Ry, gate_CNOT
import sympy as sp
import numpy as np

theta = sp.Symbol('theta', real=True)
n = 2

# Define observable
H = PauliSum.from_dict({'ZZ': 1, 'XI': 1, 'IX': 1}, n=n)

# Define circuit
circuit = [gate_Ry(theta, target=0), gate_Ry(theta, target=1), gate_CNOT(0, 1)]

# Symbolically evolve in the Heisenberg picture: U†·H·U
H_evolved = evolve(H, circuit, n_qubits=n)

# Symbolic gradient w.r.t. theta
dH = gradient(H, circuit, n_qubits=n, param_symbol=theta)

# Expectation value on a state
state = np.array([1, 0, 0, 0], dtype=complex)
ev = expectation_value(H_evolved, state, {theta: 0.5})
```

## Approximate simulation

Exact evolution grows the number of Pauli terms with circuit depth, in the worst case
until it saturates the 4ⁿ available strings. `evolve_truncated` bounds the working set
instead, discarding terms above a chosen Pauli weight after *every* gate — truncating
once at the end would save nothing, since the intermediate blow-up has already been paid
for. This is approximate Pauli-path simulation, and it is lossy by construction: the
result equals `evolve`'s only when the cutoff cannot bite.

```python
from sympauli import evolve_truncated

H_approx = evolve_truncated(H, circuit, n_qubits=n, max_weight=2)
```

A coefficient cutoff is available alongside the weight cutoff via `min_magnitude`, which
needs a `subs` dict since coefficients are symbolic; a term whose value is still unknown
after substitution is kept rather than guessed at.

Truncated evolution routes each conjugation through `conjugate_by_gate_fast`, which
recognizes a single-generator rotation `exp(-iθ/2·Q)` and applies the closed form
`G†PG = P` when `[P,Q] = 0` and `cos(θ)·P − i·sin(θ)·PQ` when `{P,Q} = 0`, instead of
building the triple product and simplifying away the terms that cancel. Anything else
falls back to the exact path, so it is a drop-in replacement for `conjugate_by_gate`.

## Package structure

| Module | Contents |
|---|---|
| `pauli_string.py` | `PauliString` — symplectic bitmask representation, multiplication, embedding |
| `pauli_sum.py` | `PauliSum` — symbolic linear combination, arithmetic, adjoint, simplification |
| `gates.py` | 40+ standard gates as `PauliSum`s: Rx/Ry/Rz, CNOT, CRy, RXX/RYY/RZZ, CCX, … |
| `heisenberg.py` | `evolve`, `gradient`, `expectation_value`, `validate` |
| `simplify.py` | `conjugate_by_gate_fast`, `as_rotation`, `simplify_coeffs`, `is_clifford_gate` |
| `truncation.py` | `evolve_truncated`, `truncate`, `truncate_weight`, `truncate_coeff` |
| `example.py` | Runnable demo: `python -m sympauli.example` |

## Qubit convention

- `PauliString.from_string('AB')`: `A` acts on qubit 1 (MSB), `B` acts on qubit 0 (LSB).
- `gate_CNOT(control=0, target=1)`: control is qubit 0, target is qubit 1.
- `to_matrix()` uses big-endian ordering: qubit `n-1` is the most significant bit.


## Citation
If you found this useful, please consider citing us.

```
@misc{shirgure2026challengesbarrenplateaumitigation,
      title={Challenges in Barren Plateau Mitigation with Dynamic Parameterized Quantum Circuits}, 
      author={Sumeet Shirgure and Efekan Kökcü and Siyuan Niu},
      year={2026},
      eprint={2606.23751},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2606.23751}, 
}
```
