# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m pytest -q                    # full suite (~3 s)
python -m pytest tests/test_gates.py   # one module
python -m pytest -s tests/test_gates.py # -s is essential: see per-assertion output (below)
python -m sympauli.example             # runnable demo of all public API
pip install -e .                       # already installed editable in this environment
```

There is no linter, formatter, or CI configured.

### Test layout is unusual

Each `tests/test_*.py` contains exactly one `test_all()` function holding hundreds of
hand-rolled `check(...)` / `check_matrix(...)` assertions that increment local `passed`/
`failed` counters and `pytest.fail()` at the end. So `pytest` reports **4 tests**, not 4
hundred, and a failure message names only the count. Always run with `-s` when
diagnosing — the individual `✗ <name>: max diff = ...` lines only reach stdout.

When adding coverage, add `check_*` calls inside the existing `test_all()` in the matching
module rather than new top-level test functions; that is the established pattern.

The canonical way to validate any new gate or evolution feature is a numeric round-trip:
`heisenberg.validate(symbolic_result, H, circuit, n, {θ: 0.4, ...})` compares
`PauliSum.to_matrix()` against `evolve_numeric()`'s dense matrix product. For a bare gate,
compare `gate.pauli_sum.to_matrix({})` to a reference NumPy matrix and check `U†U = I`.

## Architecture

Four stacked layers, each depending only on the ones below it:

| Layer | Module | Role |
|---|---|---|
| 1 | `pauli_string.py` | `PauliString`: one n-qubit Pauli, symplectic bitmask + phase |
| 2 | `pauli_sum.py` | `PauliSum`: `Σ cᵢPᵢ` with SymPy coefficients; all operator algebra |
| 3 | `gates.py` | `Gate` namedtuple; ~50 gates as local `PauliSum`s + target tuple |
| 4 | `heisenberg.py` | `evolve` / `gradient` / `expectation_value` / `validate` |

### The two representation invariants

Everything else follows from these:

1. **`PauliString` carries a phase; `PauliSum` never does.** A `PauliString` stores
   `(x_bits, z_bits, phase∈{0..3})`. Multiplying two Paulis generates a phase from the
   `_single_qubit_mul` table. When that product lands in a `PauliSum`, the phase is
   converted to `sp.I**k` and **folded into the SymPy coefficient**; the dict key is always
   the phase-free `(x_bits, z_bits, n)` triple. This is what makes `PauliSum._terms`
   canonical — exactly one entry per Pauli label, with all complex structure in the
   expression. Never write a nonzero-phase `PauliString` into `_terms`.

2. **Gates are qubit-count-agnostic until evolution time.** A `Gate` holds a `PauliSum` on
   `len(targets)` *local* qubits plus the `targets` tuple. `embed_sum(ps, targets, n_total)`
   maps local qubit `k` → global qubit `targets[k]`, and `conjugate_by_gate` calls it
   fresh on every step. So the gate library never knows the circuit width, and a gate
   object can be reused across systems of different sizes.

### Evolution mechanics

`evolve(H, circuit, n_qubits)` walks the circuit **in reverse** (`G_1` is innermost),
applying `H ← G† H G` per gate. `G†` is just `PauliSum.adjoint()` — coefficient-wise
`sp.conjugate`, valid because every Pauli string is Hermitian.

Term count multiplies at every step (`_operator_mul` is O(|A|·|B|)), so the intermediate
`simplify()` after each conjugation is load-bearing, not cosmetic — it is what prevents
expression blow-up. Passing `simplify=None` is dramatically faster per step but the
coefficients grow unboundedly. `'trig'` (default, `sp.trigsimp`) suits rotation-gate
coefficients; `'full'` (`sp.simplify`) is slow; `'expand'` is cheap and algebraic-only.

The dominant runtime cost is `_is_zero`, which calls `sp.simplify` on **every** term during
`simplify()`/`prune()`. If evolution gets slow, that is where to look first.

`gradient()` is not parameter-shift — it evolves symbolically and then `sp.diff`s each
coefficient w.r.t. the symbol.

## Qubit and label conventions

Get these wrong and results are silently incorrect, so re-derive rather than guess:

- **Labels are big-endian strings, qubit 0 is rightmost.** `PauliString.from_string('AB')`
  → `A` on qubit 1, `B` on qubit 0. `label()`/`__repr__` print the same way, and
  `to_matrix()` builds `kron` from qubit `n-1` down.
- **In `gates.py`, label positions are *local* qubit indices, mapped through `targets`.**
  For `gate_CNOT(control=0, target=1)` → `targets=(control, target)`, so local q0 = control
  and local q1 = target. Its dict `{'II','IZ','XI','XZ'}` therefore means: `'IZ'` = Z on the
  *control*, `'XI'` = X on the *target*. Same for every controlled gate — the control's
  Paulis sit in the **rightmost** label position. This trips up the derivation comments
  throughout the file.
- Rotations use the physics convention `R_P(θ) = exp(-iθ/2·P) = cos(θ/2)I - i·sin(θ/2)P`.
  Non-parameterized gates like `S`, `T`, `PhaseShift` are exact matrix decompositions
  (**not** up to global phase); `U2`/`U3`/`R` are `Rz·Ry·Rz` products and *are* only correct
  up to global phase (`test_gates.py` compares them via `up_to_global_phase`).

## Known issues

All 46 gate constructors now have a matrix reference or algebraic-relation check in
`test_gates.py`, and every gate passes a `U†U = I` sweep. Four gates were wrong before that
coverage existed — the pattern is instructive, because each bug lived in an *untested* gate
whose docstring derivation looked plausible:

- `gate_CH` had a `'ZZ'` sign error and was not unitary.
- `gate_DCX` returned the **identity**. It was built as
  `gate_CNOT(0,1).pauli_sum * gate_CNOT(1,0).pauli_sum`, but a gate's control/target
  arguments only populate its `targets` tuple — the *local* `PauliSum` is a fixed dict — so
  both factors were the same CNOT and `CNOT² = I`. **When composing gates via
  `.pauli_sum`, remember the local decomposition ignores the target arguments;** write the
  reversed-control decomposition out by hand (see the note in `gate_DCX`).
- `gate_ECR` used `(XI + YZ)/√2`, which is unitary but has the wrong sparsity pattern for
  ECR under any qubit relabeling. Correct is `(IX − XY)/√2`.
- `gate_U2` used `Rz(φ+π/2)·Ry(π/2)·Rz(λ-π/2)`, mixing the *Rx*-form's ±π/2 offsets with
  `Ry`; the result did not equal U2 even up to global phase. Correct is
  `Rz(φ)·Ry(π/2)·Rz(λ)`.

Unitarity alone is a weak check — `ECR` passed it while being the wrong gate. Always pin a
new gate to an explicit reference matrix, and add an algebraic relation
(`SISWAP² = iSWAP`, `CP(π) = CZ`, `ECR² = I`, `DCX³ = I`) where one exists.

Remaining:

- **`PauliSum.conjugate()` is an alias for `adjoint()`** and does *not* flip the sign of Y
  factors, despite its name and docstring. `adjoint_full()` is the one that applies
  `(-1)^{n_Y}` — but that is *not* the Hermitian adjoint of a Pauli sum, and nothing in the
  package uses it. Evolution correctly uses plain `adjoint()`. Treat both as suspect.
- `__init__.py` declares `__version__ = "0.1.0"` while `pyproject.toml` says `1.0.0`.
- Long docstrings in `gates.py` (`XXMinusYY`, `PSWAP`, `SISWAP`, `CCX`) contain the author's
  abandoned mid-derivation scratch work, sometimes contradicting the code beneath it. The
  code is what is numerically validated; trust the tests, not the prose.

## Context

Supporting code for arXiv:2606.23751, "Challenges in Barren Plateau Mitigation with Dynamic
Parameterized Quantum Circuits" (Shirgure, Kökcü, Niu). The README notes the package was
written in a single session with AI assistance, which is consistent with the rough edges above.
