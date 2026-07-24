# EntropyInvariant

This code is an improved nearest neighbor method for estimating differential entropy[^1] for continuous variables, invariant under change of variables, and positive. This approximation claim to solve the limiting density of discrete points formulated by Edwin Thompson Jaynes[^2]. All the details of the estimation can be found on the paper[^4].

$$
\begin{equation*}
\begin{split}
H(X)=-\int_X p(x)\log\left(\frac{p(x)}{m(x)}\right)\mathrm{d}x
\end{split}
\end{equation*}
$$

The main novelty is m(x) the invariante measure.
We introduce a proposition to describe this measure[^4] with the following properties:

$$
\begin{equation*}
\begin{aligned}
1\) & \quad m(X) = r_X > 0, \\
2\) & \quad m(aX) = am(X) = ar_X, \\
3\) & \quad m(X+b) = m(X) = r_X
\end{aligned}
\end{equation*}
$$

We found that the median value of the nearest-neighbor distance of each point is an appropriate measure for these properties.

The nearest-neighbor estimation was initialy adapted by G Varoquaux[^3] in python from this paper[^1].

In this package you can find other information theory quantities such as **joint entropy**, **conditional entropy**, **mutual information**, **conditional mutual information** and **interaction of information**. All based on a generalisation of limiting density of discrete points for more than one variable. This quantities are deduced from the chain rules (joint and relatives entropy decomposition).

Less common quantities can also be computed including **redundancy**, **information quality ratio** or **normalized mutual information**.

The methode for computing the entropy can also be specified.
- `method::String = "inv"` (optional): The method to use for entropy computation. Options are:
  - `"knn"`: k-Nearest Neighbors (k-NN) based entropy estimation.
  - `"histogram"`: Histogram-based entropy estimation.
  - `"inv"`: Invariant entropy estimation (default).

## N-source Partial Information Decomposition

`redundancy` / `unique` / `synergy` decompose `I({X,Y}; Z)` into four terms and are fixed
at two sources. `pid_lattice` gives the general N-source decomposition of Williams & Beer
(2010): `I({X_1,...,X_N}; Z)` splits into a lattice of partial-information atoms — 4 nodes
for 2 sources, 18 for 3, 166 for 4.

```julia
# continuous data, three sources
atoms = pid_lattice([x1, x2, x3], z; names = ["a", "b", "c"])
atoms["{a}{b}{c}"]   # redundancy shared by all three individually
atoms["{ab}{c}"]     # shared by the pair {a,b} and by c alone
atoms["{abc}"]       # the top synergy atom

# discrete data, from an explicit joint distribution (target in the last dimension).
# Two-input AND, reproducing Williams & Beer's published values in bits:
pmf = zeros(2, 2, 2)
pmf[1,1,1] = pmf[1,2,1] = pmf[2,1,1] = 0.25; pmf[2,2,2] = 0.25
a = pid_lattice(pmf; names = ["X", "Y"])
a["{X}{Y}"]   # 0.3113  (redundancy)
a["{XY}"]     # 0.5     (synergy)
```

Three redundancy measures are available:

- **`:mmi`** (default, continuous data) — `min` over coalitions of `I(X_A; Z)`.
  Works with any of the package's estimators and reduces exactly to `redundancy` /
  `unique` at two sources.
- **`:imin`** (discrete data, via `pid_lattice(pmf)`) — Williams & Beer's original measure,
  using specific information per target outcome. Guarantees non-negative atoms, which
  `:mmi` does not for three or more sources. Over-credits redundancy: on two-bit COPY it
  reports two *independent* bits as fully redundant.
- **`:iccs`** (discrete data) — Ince's (2017) pointwise common change in surprisal. Fixes
  the COPY case (`R = 0`, `U_X = U_Y = 1`, `Syn = 0`) and, unlike `:mmi`, lets both unique
  atoms be positive at once. Atoms may be negative; Ince argues these are meaningful.

```julia
# the same distribution under two measures — Z = (X, Y), independent bits
pmf = zeros(2, 2, 4)
pmf[1,1,1] = pmf[1,2,2] = pmf[2,1,3] = pmf[2,2,4] = 0.25
pid_lattice(pmf; measure = :imin, names = ["X","Y"])["{X}{Y}"]   # 1.0  — over-credited
pid_lattice(pmf; measure = :iccs, names = ["X","Y"])["{X}{Y}"]   # 0.0  — correct
```

### Two things worth knowing before interpreting the output

**Estimated coalition MIs need repairing first.** True mutual information satisfies
`I(X_A; Z) ≤ I(X_B; Z)` whenever `A ⊆ B`, but finite-sample kNN estimates measurably do
not — adding a source that is nearly redundant with the existing ones *lowers* the
estimate. Both the `min` in `:mmi` and the Möbius inversion assume that ordering, so
unrepaired estimates produce negative unique and synergy atoms, which are impossible for
true information. `pid_lattice` therefore applies `isotonic_repair` by default
(`repair = :isotonic`, or `:majorant` / `:none`). The repair deliberately does **not**
clamp to zero: an estimate of `-0.02` where the truth is near zero is ordinary symmetric
noise, and truncating it would bias every low-signal region upward.

**`:mmi`'s unique atoms are winner-take-all.** At two sources they are
`max(0, I_X - I_Y)` and `max(0, I_Y - I_X)`, so exactly one is nonzero by construction.
That is appropriate for asking *how information is divided*, but it cannot express
"partly one source, partly the other" and will not move gradually as an underlying
relationship shifts. For *how much each source contributes*, use
`conditional_mutual_information`, which is unclipped and continuous.

Note also that `sum(atoms) == I({all sources}; Z)` holds identically for any redundancy
measure — it is a property of the Möbius inversion, not a check that the decomposition is
right. Correctness is established in the test suite against published atom values on
discrete toy distributions (AND, XOR, two-bit COPY, three-way XOR), the Williams & Beer
non-negativity theorem, and exact agreement with `redundancy` / `unique` at two sources.


## Package Structure

The package follows Julia best practices with a modular organization:

```
src/
├── EntropyInvariant.jl          # Main module (exports, includes)
├── types.jl                     # Data structures (DataShape, KNNResult)
├── helpers/
│   ├── utility_helpers.jl       # Utility functions (nn1, histograms, logging)
│   ├── data_helpers.jl          # Data transformation & validation
│   └── computation_helpers.jl   # Core computation helpers
├── entropy.jl                   # Entropy estimation functions
├── mutual_information.jl        # Mutual information & conditional entropy
├── advanced.jl                  # Advanced information theory functions
├── pid.jl                       # Partial Information Decomposition (2 sources)
├── pid_lattice.jl               # N-source PID: Williams & Beer redundancy lattice
└── optimized.jl                 # Optimized matrix computations
```

This modular structure improves code maintainability while preserving all functionality and numerical accuracy.

**Example usage**:

```julia
n = 1000
k = 3
p1 = rand(n)

println("Entropy invariant:")
println(entropy(p1, k=k, verbose=true))
println(entropy(1e5*p1.-123.465, k=k))
println(entropy(1e-5*p1.+654.321, k=k))

p2 = rand(n)
p3 = rand(n)           

println("\nJoint entropy invariant:")
println(EntropyInvariant.entropy(hcat(p1,p2,p3), verbose=true))
println(entropy(hcat(p1, 1e5*p2.-123.456, 1e-5*p3.+654.123)))

p1 = rand(n)
p2 = 2*p1+rand(n)

println("\nMutual Information invariant: ")
println(mutual_information(p1, p2, k=k))
println(mutual_information(1e5*p1.+123.456, 1e-5*p2.+654.321, k=k))

# Using k-NN method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (k-NN): ", entropy(data, method="knn", k=5, verbose=true))

# Using histogram method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (Histogram): ", entropy(data, method="histogram", nbins=10) )

# Using invariant method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (Invariant): ", entropy(data, method="inv", k=3))

```

In extreme cases, when the neighbourhood distance is small. The logarithm of the distance is strongly negative. This can lead to negative entropy. We therefore recommend setting the "degenerate" parameter to true. This parameter adds 1 to each distance, so that the logarithm is always positive.

Please inform us if you discover any bugs or errors in the code, or if you believe another quantity should be added.

[^1]: [Estimating mutual information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138) DOI: 10.1103/PhysRevE.69.066138
[^2]: [Prior Probabilities](https://ieeexplore.ieee.org/document/4082152) DOI: 10.1109/TSSC.1968.300117
[^3]: [Python git](https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429) G Varoquaux
[^4]: [An Invariant Measure for Differential Entropy: From Kullback–Leibler Divergence to Scale-Invariant Information Theory](https://www.mdpi.com/1099-4300/28/3/301) DOI: 10.3390/e28030301

**AUTHORS:** Félix TRUONG, Alexandre GIULIANI

**References:**
