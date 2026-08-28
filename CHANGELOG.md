# Changelog

All notable changes to this project are documented in this file, starting
from this release.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.2.2] - 2026-08-28

### Changed
- `[compat]` now declares `julia = "1.10"` instead of `"1.1"`. The old bound was never
  achievable: `SpecialFunctions` 2.6 and later require Julia 1.10, and even 2.5 requires
  1.5, so no valid dependency set resolves on Julia 1.1. 1.10 is the current LTS and is
  the oldest version CI tests. No source change -- this only makes the manifest describe
  what the package actually supports.

## [2.2.1] - 2026-08-14

### Fixed
- KSG/Frenzel-Pompe neighbour counting used a fixed absolute epsilon (`1e-12`) to turn
  `inrangecount`'s non-strict (`<=`) radius comparison into the strict (`<`) one the
  estimators require. Any genuine neighbour lying within that epsilon of the shared
  radius was dropped along with the k-th one, so the marginal counts came out too low.
  The correction is now relative -- `prevfloat`, exactly one ULP -- so it scales with the
  radius instead of assuming one. Invariant normalization keeps *typical* distances near
  1, which is what made the absolute epsilon look safe, but it cannot keep individual
  neighbours away from the radius: data mixing two very different scales (a cluster
  orders of magnitude tighter than the median spacing, alongside a normal spread) puts
  many neighbours inside that window at once. `mutual_information` and
  `conditional_mutual_information` under `method="inv_ksg"` were affected, and with them
  every PID atom built on them. Measured on a 70%-tight-core mixture with a true MI of 0,
  the estimate moved from -0.099 to +0.001 nats. Tie-free data is unchanged.
- A degenerate shared radius (`k+1` coincident points) reported
  `ArgumentError: the query radius r must be >= 0` from NearestNeighbors instead of the
  package's own message describing the data. `inrangecount` rejects a negative radius
  outright, so the degenerate case is now caught from the radius itself, before the
  query. This is the one place the Julia and Python implementations differ: SciPy's
  `query_ball_point` accepts a negative radius and returns a count of 0, so the Python
  side can detect it after the fact.

## [2.2.0] - 2026-07-25

### Added
- `pid_lattice`: N-source Partial Information Decomposition over the Williams & Beer
  (2010) redundancy lattice, generalizing the two-source `redundancy`/`unique`/`synergy`
  triple. 4 atoms for 2 sources, 18 for 3, 166 for 4. Two redundancy measures: `:mmi`
  (min over coalition mutual informations, estimable from continuous data with any of the
  package's estimators, and exactly reducing to `redundancy`/`unique` at two sources) and
  `:imin` (Williams & Beer's original specific-information measure, on an explicit
  discrete joint distribution, with guaranteed non-negative atoms) and `:iccs`
  (Ince 2017, pointwise common change in surprisal).
- `iccs_redundancy`: Ince's `I_ccs`. Keeps only the pointwise co-information whose sign
  every coalition agrees on. Corrects the two-bit COPY case that `:imin` gets wrong
  (`R = 0`, `U_X = U_Y = 1`, `Syn = 0` rather than `R = 1`), and unlike `:mmi` allows both
  unique atoms to be positive simultaneously instead of exactly one by construction.
  Defined for any number of sources, unlike BROJA and other optimisation-based measures.
  Not guaranteed monotone on the lattice, so atoms may be negative.
- `redundancy_lattice`, `lattice_labels`, `moebius_atoms`: the lattice structure,
  human-readable atom labels, and the Möbius inversion, exposed separately so a custom
  redundancy measure can be plugged in.
- `coalition_mutual_information`: `I(X_A; Z)` for every coalition of sources. Uses the
  dimension-agnostic shared-radius KSG estimator for `method="inv_ksg"`; rejects
  coalitions beyond 3 total dimensions for the plug-in methods with an explicit message
  rather than an opaque failure from deeper in the stack.
- `isotonic_repair`: projects estimated coalition mutual informations onto the monotone
  cone `I(X_A;Z) <= I(X_B;Z)` for `A` a subset of `B`. Finite-sample kNN estimates violate
  this — adding a nearly-redundant source lowers the estimate — which otherwise
  manufactures negative unique and synergy atoms that are impossible for true information.
  Deliberately does not clamp to zero, since non-negativity constrains the level of a
  single estimate rather than the consistency between two, and clamping would bias
  low-signal regions upward.
- `specific_information`: Williams & Beer's per-target-outcome specific information.
- 171 tests covering the above: lattice node counts (4/18/166) and rejection beyond 4
  sources, published atom values for AND / XOR / two-bit COPY / three-way XOR, the
  Williams & Beer non-negativity theorem over 100 random three-source distributions, exact
  agreement with `redundancy`/`unique` at two sources, the monotonicity repair, and
  `I_ccs` against exactly-derived values (AND redundancy is `0.25*log2(4/3)`).

## [2.1.0] - 2026-07-24

### Added
- `MI()` / `CMI()` (`method="inv_ksg"`) now build each dimension's own k-NN
  tree once and reuse it across all pairs, instead of rebuilding it (and, for
  `CMI()`, the Z-only tree) redundantly for every pair -- a pure speed
  improvement, no change in output values.
- `parallel::Bool = false` keyword on `MI()` / `CMI()`: distributes the
  remaining O(m²) per-pair shared-radius work across `Threads.@threads`
  (requires Julia started with more than one thread to actually parallelize).
  Verified to produce identical results to `parallel=false`.

## [2.0.0] - 2026-07-24

Bumped as a major version because the default method now returns different
numeric values than before, even though no function signatures changed.
Mirrors the same release in the companion Python package
([entropy-invariant](https://github.com/Entropy-Invariant/entropy-invariant)).

### Added
- `mutual_information_ksg` / `conditional_mutual_information_ksg`: a KSG
  (Kraskov, Stögbauer & Grassberger 2004) / Frenzel-Pompe (2007) shared-radius
  estimator, applied after invariant-measure normalization. Cancels the
  leading-order k-NN bias that the plug-in formula (`H(X)+H(Y)-H(X,Y)`, etc.)
  does not, most visibly on outlier-contaminated or near-degenerate data.
- `method="inv_ksg"` is now the option throughout `mutual_information`,
  `conditional_mutual_information`, `conditional_entropy`,
  `normalized_mutual_information`, `interaction_information`,
  `information_quality_ratio`, `redundancy`, `unique`, `synergy`, and the
  matrix fast-paths `MI()` / `CMI()`.
- Expanded docs: `docs/src/getting_started.md`, `docs/src/tutorial.md` (new
  outlier-robustness example), `docs/src/api.md`.

### Changed
- **Breaking**: the default `method` for all MI/CMI-derived functions changed
  from `"inv"` (plug-in) to `"inv_ksg"`. `method="inv"` is still available
  and unchanged. Anything relying on the default now gets different (more
  bias-corrected) numeric output.

### Fixed
- `MI()` / `CMI()` (the matrix fast-path) computed the invariant measure
  inline without filtering zero values, unlike the documented
  `compute_invariant_measure()` helper. On sparse data (mostly zeros, e.g.
  real spectral/sensor data) this produced a zero median nearest-neighbor
  distance, causing division by zero and NaN/Inf propagation instead of
  matching the scalar `mutual_information()` / `conditional_mutual_information()`
  functions.
- `compute_knn_entropy_nats` used the *post-filtering* sample count (after
  dropping points with a degenerate zero k-th-neighbor distance) instead of
  the true total number of points for the `digamma(n)` term. These agree for
  continuous data with no ties, but diverge sharply on data with duplicates —
  up to several nats of error on realistic sparse test data.
- The KSG/Frenzel-Pompe shared-radius estimator silently produced `NaN`
  (`digamma(0) = -Inf`) when ≥k+1 points coincide exactly in the joint space
  it searches over. Now throws a clear `ArgumentError`. `MI()`'s diagonal and
  `mutual_information_ksg(x, x)` (i.e. `I(X;X) = H(X)`) no longer run the
  shared-radius trick at all, since pairing a variable with itself makes this
  case trivial to hit for any column with duplicate values.
- `compute_invariant_measure()` now throws a clear `ArgumentError` when the
  invariant measure is degenerate (too many duplicate non-zero values),
  instead of silently returning a value that produces a cryptic downstream
  crash.

## Earlier releases

Versions [v1.1.1], [v1.1.0], and [v1.0.0] predate this changelog; see their
GitHub release pages for the (auto-generated, Julia-registry-compat-only)
notes.

[2.2.0]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v2.2.0
[2.1.0]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v2.1.0
[2.0.0]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v2.0.0
[v1.1.1]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v1.1.1
[v1.1.0]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v1.1.0
[v1.0.0]: https://github.com/Entropy-Invariant/EntropyInvariant.jl/releases/tag/v1.0.0
