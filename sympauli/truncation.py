"""
truncation.py
-------------
Approximate Pauli-path simulation for the Symbolic Pauli Heisenberg Evolution
Engine.

Exact Heisenberg evolution is exact and, in the worst case, exponentially
expensive: every conjugation by a rotation can double the number of Pauli
strings, so the term count grows with circuit depth until it saturates the 4ⁿ
strings available.  Approximate Pauli-path simulation trades that blow-up for a
controlled error by discarding the terms that are expected to contribute least
— those of high Pauli weight, whose coefficients are typically the smallest,
and those whose coefficient magnitude is directly measurable as small.

The two knobs are independent:

    truncate_weight(ps, max_weight)          drop terms with weight > max_weight
    truncate_coeff(ps, min_magnitude, subs)  drop terms with |c| < min_magnitude

Weight truncation is the cheap and principled one.  `PauliString.weight` is
popcount(x_bits | z_bits), so filtering by it touches no SymPy at all — it is a
pure filter over the `_terms` keys — and it is exactly the cutoff the Pauli-path
literature analyses.  Magnitude truncation needs numbers, and coefficients here
are symbolic, so it takes a `subs` dict; see the conservative rule documented on
that function.

----------------------------------------------------------------------
Where truncation happens, and why the order matters
----------------------------------------------------------------------

`evolve_truncated` truncates after *every* gate.  Truncating once at the end
saves nothing: the intermediate blow-up has already been paid for, both in the
operator products and in the SymPy simplification of every term produced along
the way.  Bounding the working set at each step is what makes the method cheap,
and is what "approximate Pauli-path simulation" means.

Within one step the order is **conjugate → simplify → truncate**.  Truncation
does not commute with simplification in general: simplification merges the
terms that share a Pauli string, and a merge can lift two sub-threshold
coefficients above the magnitude cutoff, or cancel two above-threshold ones
down to zero.  Simplifying first means (a) magnitude checks see a coefficient
in canonical form rather than an unreduced sum of contributions, and (b) weight
filtering acts on the final, merged term set.  The opposite order would discard
terms whose contribution had not yet been accumulated.

Note that truncation makes the result an approximation, and one whose error is
not tracked here.  A truncated evolution is not `evolve`'s output with fewer
terms; it is the output of a different, lossy map.  The one case where it
coincides exactly is a cutoff that cannot bite — `max_weight >= n_qubits` with
no magnitude cutoff — which `tests/test_truncation.py` pins down.

----------------------------------------------------------------------
Public API
----------------------------------------------------------------------

    truncate_weight(ps, max_weight)                       -> PauliSum
    truncate_coeff(ps, min_magnitude, subs=None)          -> PauliSum
    truncate(ps, max_weight=None, min_magnitude=None, subs=None) -> PauliSum
    evolve_truncated(observable, circuit, n_qubits, ...)   -> PauliSum
"""

from __future__ import annotations

from typing import Sequence

from .pauli_sum import PauliSum
from .gates import Gate
from .heisenberg import conjugate_by_gate
from .simplify import conjugate_by_gate_fast, simplify_coeffs


# ---------------------------------------------------------------------------
# Weight truncation
# ---------------------------------------------------------------------------

def truncate_weight(ps: PauliSum, max_weight: int) -> PauliSum:
    """
    Drop every term whose Pauli weight exceeds `max_weight`.

    The weight of a Pauli string is its number of non-identity factors,
    popcount(x_bits | z_bits), so this is a filter over dict keys with no
    SymPy involved — deliberately, since it runs after every gate.

    Returns a new PauliSum; `ps` is untouched.
    """
    result = PauliSum(ps.n)
    for key, coeff in ps._terms.items():
        x_bits, z_bits, _ = key
        if bin(x_bits | z_bits).count('1') <= max_weight:
            result._terms[key] = coeff
    return result


# ---------------------------------------------------------------------------
# Magnitude truncation
# ---------------------------------------------------------------------------

def truncate_coeff(
    ps: PauliSum,
    min_magnitude: float,
    subs: dict | None = None,
) -> PauliSum:
    """
    Drop terms whose coefficient magnitude is below `min_magnitude`.

    Coefficients are SymPy expressions, so a magnitude only exists once the
    free symbols have values: `subs` is applied to a *copy* of each coefficient
    purely to make the decision, and the term that survives keeps its original
    symbolic coefficient.  Truncation removes terms; it never substitutes.

    If a coefficient still has free symbols after `subs` — or resists
    conversion to a complex number — the term is **kept**.  A coefficient whose
    value is unknown cannot be judged small, and wrongly keeping a negligible
    term costs a little work while wrongly dropping a large one silently
    corrupts the result.  A consequence worth stating plainly: calling this
    with no `subs` on a fully symbolic PauliSum is a no-op.

    Returns a new PauliSum; `ps` is untouched.
    """
    result = PauliSum(ps.n)
    for key, coeff in ps._terms.items():
        probe = coeff.subs(subs) if subs else coeff
        if probe.free_symbols:
            result._terms[key] = coeff
            continue
        try:
            magnitude = abs(complex(probe.evalf()))
        except (TypeError, ValueError):
            result._terms[key] = coeff
            continue
        if magnitude >= min_magnitude:
            result._terms[key] = coeff
    return result


def truncate(
    ps: PauliSum,
    max_weight: int | None = None,
    min_magnitude: float | None = None,
    subs: dict | None = None,
) -> PauliSum:
    """
    Apply weight truncation and then magnitude truncation, skipping whichever
    cutoff is None.  With both None this returns a copy of `ps`.

    Weight goes first because it is free and shrinks the set the (SymPy-bound)
    magnitude pass has to walk.
    """
    result = ps
    if max_weight is not None:
        result = truncate_weight(result, max_weight)
    if min_magnitude is not None:
        result = truncate_coeff(result, min_magnitude, subs)
    return result if result is not ps else ps.copy()


# ---------------------------------------------------------------------------
# Truncated evolution
# ---------------------------------------------------------------------------

def evolve_truncated(
    observable: PauliSum,
    circuit: Sequence[Gate],
    n_qubits: int,
    max_weight: int | None = None,
    min_magnitude: float | None = None,
    subs: dict | None = None,
    simplify: str | None = "trig",
    use_fast: bool = True,
    verbose: bool = False,
) -> PauliSum:
    """
    Heisenberg evolution with truncation applied after each gate conjugation:

        H ← truncate(simplify(G† H G))   for G = G_n, ..., G_1

    Gates are walked in reverse exactly as in `heisenberg.evolve` (G_1 is the
    innermost, applied first), and the per-step cost is bounded by the number of
    terms that survive truncation rather than by the number the exact evolution
    would have accumulated.

    Parameters
    ----------
    observable    : PauliSum on n_qubits qubits — the initial observable H
    circuit       : ordered gate sequence [G_1, G_2, ..., G_n]
    n_qubits      : total number of qubits
    max_weight    : Pauli-weight cutoff, or None for no weight truncation
    min_magnitude : coefficient-magnitude cutoff, or None
    subs          : {symbol: value} used to evaluate magnitudes (see
                    `truncate_coeff`); ignored when min_magnitude is None
    simplify      : method applied after each conjugation, resolved through
                    `simplify_coeffs` (so 'cancel' and 'none' are available on
                    top of 'trig' / 'full' / 'expand'), or None to skip it
    use_fast      : route each step through `simplify.conjugate_by_gate_fast`,
                    which uses the closed form for rotation gates and the exact
                    product for everything else.  False forces the exact path
                    for every gate; the two agree up to simplification.  Only
                    the conjugation changes — simplification is identical
                    either way, so this flag isolates the closed form.
    verbose       : print the term count after each gate, before and after
                    truncation

    Returns
    -------
    PauliSum — the truncated evolution of H.  Exact only when the cutoffs
    cannot bite (e.g. max_weight >= n_qubits and min_magnitude None).
    """
    H = observable

    if verbose:
        print(f"Initial observable: {len(H)} term(s)")

    def step(H_in: PauliSum, gate: Gate) -> PauliSum:
        if use_fast:
            return conjugate_by_gate_fast(H_in, gate, n_qubits, simplify=simplify)
        exact = conjugate_by_gate(H_in, gate, n_qubits, simplify=None)
        return simplify_coeffs(exact, simplify) if simplify else exact

    for i, gate in enumerate(reversed(circuit)):
        H = step(H, gate)
        n_before = len(H)
        H = truncate(H, max_weight, min_magnitude, subs)
        if verbose:
            print(f"  After gate {len(circuit)-i}: {gate.name:25s}  →  "
                  f"{n_before} term(s), {len(H)} after truncation")

    return H
