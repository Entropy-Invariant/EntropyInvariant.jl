# Tutorial

This tutorial provides detailed examples of using EntropyInvariant.jl for information-theoretic analysis.

## Example 1: Measuring Dependence Between Variables

Mutual information measures how much information two variables share. It's zero for independent variables and positive for dependent ones.

```julia
using EntropyInvariant

n = 5000

# Independent variables
x = rand(n)
y = rand(n)
I_independent = mutual_information(x, y)
println("Independent variables: I(X;Y) = $I_independent")  # Close to 0

# Linearly dependent variables
x = rand(n)
y = 2*x + 0.1*rand(n)
I_linear = mutual_information(x, y)
println("Linear dependence: I(X;Y) = $I_linear")  # Positive

# Nonlinear dependence
x = rand(n) .- 0.5
y = x.^2 + 0.1*rand(n)
I_nonlinear = mutual_information(x, y)
println("Nonlinear dependence: I(X;Y) = $I_nonlinear")  # Also positive!
```

**Note**: Unlike correlation, mutual information captures both linear and nonlinear dependencies.

## Example 2: Information Chain Rules

Information theory provides elegant decomposition rules. Let's verify them.

These identities are exact for the plug-in estimator (`method="inv"`), since
`conditional_entropy` and `mutual_information` are algebraically *defined* as differences
of the same `entropy()` calls under that method. The default `method="inv_ksg"` instead
estimates I(X;Y) and H(Y|X) directly with a different (more accurate) estimator, so
these identities hold only approximately under the default -- see Example 4 below for
why that trade is usually worth it.

### Chain Rule for Entropy

$H(X,Y) = H(X) + H(Y|X)$

```julia
using EntropyInvariant

n = 5000
x = rand(n)
y = rand(n)

H_X = entropy(x)
H_Y_given_X = conditional_entropy(x, y, method="inv")  # conditional_entropy(X, Y) = H(Y|X)
H_XY = entropy(hcat(x, y))

println("H(X) = $H_X")
println("H(Y|X) = $H_Y_given_X")
println("H(X) + H(Y|X) = $(H_X + H_Y_given_X)")
println("H(X,Y) = $H_XY")
# These should be approximately equal!
```

### Mutual Information via Entropy

$I(X;Y) = H(X) + H(Y) - H(X,Y)$

```julia
n = 5000
x = rand(n)
y = x + 0.5*rand(n)

H_X = entropy(x)
H_Y = entropy(y)
H_XY = entropy(hcat(x, y))
I_XY = mutual_information(x, y, method="inv")

println("H(X) + H(Y) - H(X,Y) = $(H_X + H_Y - H_XY)")
println("I(X;Y) = $I_XY")
# These should be approximately equal!
```

## Example 3: Conditional Mutual Information

Conditional MI measures the dependence between X and Y given knowledge of Z:

$I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)$

```julia
using EntropyInvariant

n = 5000

# Z causes both X and Y (confounding variable)
z = rand(n)
x = z + 0.3*rand(n)
y = z + 0.3*rand(n)

I_XY = mutual_information(x, y)
I_XY_given_Z = conditional_mutual_information(x, y, z)

println("I(X;Y) = $I_XY")          # High: X and Y appear dependent
println("I(X;Y|Z) = $I_XY_given_Z") # Low: controlling for Z removes dependence
```

## Example 4: Why not just use `std` normalization? A few extreme outliers

Every function so far normalized each variable by its **invariant measure** (median
nearest-neighbor distance) before running a shared-radius KSG search -- rather than the
`std`-based normalization used by most other MI/CMI implementations. For *moderate*
pathologies (skew, heavy tails, a fraction of moderate-sized outliers), it barely
matters: both normalizations, followed by a shared-radius KSG estimator, give close
results.

Where they diverge sharply is a **small number of extreme values**. `std` is a
variance-based statistic that a single sufficiently extreme point can dominate; the
invariant measure barely notices, since the median is robust by construction. We can
demonstrate this directly by reusing the package's internal KSG core
(`EntropyInvariant._mi_ksg_from_normalized`) with `std`-based normalization instead of
the invariant measure, and comparing the two side by side:

```julia
using EntropyInvariant
using Statistics

# MI computed after std-normalizing instead of invariant-measure-normalizing --
# the same shared-radius KSG algorithm, different normalization constant.
function mi_std(x, y; k=3)
    xs = reshape((x .- mean(x)) ./ std(x), 1, :)
    ys = reshape((y .- mean(y)) ./ std(y), 1, :)
    return EntropyInvariant._mi_ksg_from_normalized(Matrix{Float64}(xs), Matrix{Float64}(ys), k)
end

n = 2000
x_clean = randn(n)
y = 0.8*x_clean + 0.3*randn(n)   # true dependency, ~1.05 nats

true_mi = mutual_information_ksg(x_clean, y)
println("True MI (no contamination): $true_mi")

# Knob 1: how many extreme points (fixed magnitude = 1e4)?
println("\ncount | inv_ksg | std-normalized")
for n_out in [0, 1, 2, 5, 10, 20, 50]
    x = copy(x_clean)
    if n_out > 0
        idx = rand(1:n, n_out)
        x[idx] .= 1e4
    end
    println("$n_out | $(mutual_information_ksg(x, y)) | $(mi_std(x, y))")
end

# Knob 2: how extreme (fixed count = 5 points)?
println("\nmagnitude | inv_ksg | std-normalized")
for mag in [1, 10, 100, 1000, 1e4, 1e5]
    x = copy(x_clean)
    idx = rand(1:n, 5)
    x[idx] .= mag
    println("$mag | $(mutual_information_ksg(x, y)) | $(mi_std(x, y))")
end
```

Two things to notice: **count matters a lot** -- even a couple of points out of 2000
start pulling std-normalized MI down -- and **magnitude has a threshold, not a gradual
slope**: std-normalized MI stays fine while the extra points are only mildly larger than
the bulk, then drops sharply once they're extreme enough to dominate the variance
calculation. `mutual_information_ksg` (invariant-measure-normalized) stays flat across
the entire range in both sweeps.

This -- not moderate skew or moderate outlier fractions -- is the practical scenario
where this package's normalization has a real, measurable edge: a handful of sensor
glitches, data-entry errors, or rare extreme events sitting in otherwise well-behaved
data.

## Example 5: Normalized Mutual Information

NMI normalizes mutual information to the range [0, 1], useful for comparing across datasets:

```julia
using EntropyInvariant

n = 5000

# Different levels of dependence
noise_levels = [0.01, 0.1, 0.5, 1.0, 2.0]

println("Noise Level | MI        | NMI")
println("-" ^ 35)

for noise in noise_levels
    x = rand(n)
    y = x + noise * rand(n)

    mi = mutual_information(x, y)
    nmi = normalized_mutual_information(x, y)

    println("$noise         | $(round(mi, digits=3))    | $(round(nmi, digits=3))")
end
```

## Example 6: Interaction Information

Interaction information (also called co-information) measures three-way interactions:

- Positive: X, Y, and Z share common information (redundancy dominates)
- Negative: X and Y together tell more about Z than separately (synergy dominates)

```julia
using EntropyInvariant

n = 5000

# XOR-like relationship (synergistic)
x = rand(n) .> 0.5
y = rand(n) .> 0.5
z = Float64.(xor.(x, y)) + 0.1*rand(n)

I_interaction = interaction_information(Float64.(x), Float64.(y), z)
println("XOR relationship (synergy): II = $I_interaction")  # Negative

# Common cause (redundant)
z = rand(n)
x = z + 0.1*rand(n)
y = z + 0.1*rand(n)

I_interaction = interaction_information(x, y, z)
println("Common cause (redundancy): II = $I_interaction")  # Positive
```

## Example 7: Partial Information Decomposition

PID decomposes mutual information into redundancy, unique information, and synergy:

$I(X,Y;Z) = R(X,Y;Z) + U(X;Z) + U(Y;Z) + S(X,Y;Z)$

```julia
using EntropyInvariant

n = 5000

# Z = X + Y + noise
x = rand(n)
y = rand(n)
z = x + y + 0.1*rand(n)

R = redundancy(x, y, z)
U_x, U_y = EntropyInvariant.unique(x, y, z)
S = synergy(x, y, z)

println("Redundancy R(X,Y;Z) = $R")
println("Unique U(X;Z) = $U_x")
println("Unique U(Y;Z) = $U_y")
println("Synergy S(X,Y;Z) = $S")
println()
println("Total: $(R + U_x + U_y + S)")
println("I(X,Y;Z) = $(mutual_information(hcat(x,y), reshape(z, n, 1)))")
```

## Example 8: Optimized Matrix Functions

For computing pairwise MI or CMI across many variables, use the optimized matrix functions:

```julia
using EntropyInvariant

n = 1000
m = 5  # 5 variables

# Random data matrix
data = rand(n, m)

# Compute pairwise MI matrix (much faster than loops)
MI_matrix = EntropyInvariant.MI(data)

println("Pairwise Mutual Information Matrix:")
display(round.(MI_matrix, digits=3))

# Conditional MI given another variable
z = rand(n)
CMI_matrix = EntropyInvariant.CMI(data, z)

println("\nPairwise CMI given Z:")
display(round.(CMI_matrix, digits=3))
```

## Example 9: Comparing Estimation Methods

```julia
using EntropyInvariant

n = 5000
x = randn(n)  # Standard normal: theoretical entropy = 0.5*log(2πe) ≈ 1.42

H_inv = entropy(x, method="inv")
H_knn = entropy(x, method="knn")
H_hist = entropy(x, method="histogram", nbins=50)

theoretical = 0.5 * log(2 * π * exp(1))

println("Theoretical (normal): $theoretical")
println("Invariant method:     $H_inv")
println("k-NN method:          $H_knn")
println("Histogram method:     $H_hist")
```

## Example 10: Effect of Sample Size and k

```julia
using EntropyInvariant

# True entropy of uniform distribution on [0,1] is 0
true_H = 0.0

println("Sample Size | k=3      | k=5      | k=10")
println("-" ^ 45)

for n in [100, 500, 1000, 5000]
    x = rand(n)

    H3 = entropy(x, k=3)
    H5 = entropy(x, k=5)
    H10 = entropy(x, k=10)

    println("$n        | $(round(H3, digits=3))   | $(round(H5, digits=3))   | $(round(H10, digits=3))")
end
```

## Example 11: Multi-dimensional Joint Entropy

```julia
using EntropyInvariant

n = 2000

# Joint entropy of independent variables
d = 3
X_independent = rand(n, d)
H_independent = entropy(X_independent)

# Should be approximately d * H(uniform) = d * 0 = 0
println("Joint entropy of $d independent uniforms: $H_independent")

# Sum of marginal entropies
H_marginals = sum(entropy(X_independent[:, i]) for i in 1:d)
println("Sum of marginal entropies: $H_marginals")
# For independent variables, these should be equal
```

## Tips and Best Practices

1. **Sample size**: Use at least 500-1000 points for reliable estimates
2. **k parameter**: Default k=3 works well; increase for noisy data
3. **Degenerate data**: Use `degenerate=true` for data with ties or small distances
4. **Matrix functions**: Use `MI()` and `CMI()` for pairwise computations (much faster)
5. **Base conversion**: Use `base=2` for bits, `base=e` (default) for nats
