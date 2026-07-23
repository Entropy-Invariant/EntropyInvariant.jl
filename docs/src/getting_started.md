# Getting Started

## Installation

EntropyInvariant.jl can be installed using Julia's package manager:

```julia
using Pkg
Pkg.add("EntropyInvariant")
```

Or from the Julia REPL package mode (press `]`):

```
pkg> add EntropyInvariant
```

## Basic Usage

### Loading the Package

```julia
using EntropyInvariant
```

### Computing Entropy

The simplest use case is computing the entropy of a random variable:

```julia
# Generate random data
x = rand(1000)

# Compute entropy using the invariant method (default)
H = entropy(x)
println("Entropy: $H")
```

### Data Organization

EntropyInvariant supports two data layouts controlled by the `dim` parameter:

- **`dim=1` (default)**: Each row is a data point, each column is a dimension
- **`dim=2`**: Each column is a data point, each row is a dimension

```julia
n = 1000  # number of points
d = 3     # number of dimensions

# Row-oriented data (default)
data_rows = rand(n, d)  # 1000 points × 3 dimensions
H1 = entropy(data_rows)  # or entropy(data_rows, dim=1)

# Column-oriented data
data_cols = rand(d, n)  # 3 dimensions × 1000 points
H2 = entropy(data_cols, dim=2)

# Both give the same result
```

### Choosing an Estimation Method

Three methods are available via the `method` parameter:

```julia
x = rand(1000)

# Invariant method (default) - scale and translation invariant
H_inv = entropy(x, method="inv")

# Standard k-NN method
H_knn = entropy(x, method="knn")

# Histogram method (only for 1-3 dimensions)
H_hist = entropy(x, method="histogram", nbins=20)
```

### Common Parameters

Most functions accept these optional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3 | Number of nearest neighbors |
| `base` | `e` | Logarithm base (e, 2, 10, etc.) |
| `method` | `"inv"` for `entropy`; `"inv_ksg"` for MI/CMI-derived functions | See below |
| `dim` | 1 | Data layout (1=rows as points, 2=columns as points) |
| `verbose` | `false` | Print computation details |
| `degenerate` | `false` | Handle degenerate cases (adds 1 to distances). Ignored by `"inv_ksg"`. |

### Two normalizations, two algorithms

Every quantity in this package starts by normalizing each variable by its **invariant
measure** (median nearest-neighbor distance) rather than `std` -- that normalization is
what buys scale/translation invariance and, unlike `std`, doesn't get thrown off by a
handful of extreme outliers.

For a single variable's entropy, there is only one sensible algorithm to run *after*
normalizing (Kozachenko-Leonenko). But mutual information and everything built on it
(`mutual_information`, `conditional_mutual_information`, `conditional_entropy`,
`normalized_mutual_information`, `interaction_information`, `information_quality_ratio`,
`redundancy`/`unique`/`synergy`, and the `MI`/`CMI` matrices) can be computed two ways
once the data is normalized:

- **naively differencing entropies** (`method="inv"`) -- `I(X;Y) = H(X) + H(Y) - H(X,Y)`,
  three independent k-NN searches, each with its own finite-sample bias
- **KSG / Frenzel-Pompe** (`method="inv_ksg"`, the **default**) -- one shared k-NN search
  radius reused across all terms, so the leading-order bias cancels algebraically instead
  of compounding

`method="inv_ksg"` is both more accurate and the default for all of these functions.
`method="inv"` remains available when you specifically need the individual entropy terms
(e.g. `H(X)`, `H(X,Y)` separately), not just their difference.

### Computing Mutual Information

```julia
n = 1000
x = rand(n)
y = 2*x + 0.1*rand(n)  # y depends on x

# Mutual information (method="inv_ksg" by default)
I_xy = mutual_information(x, y)
println("I(X;Y) = $I_xy")

# With different parameters
I_xy = mutual_information(x, y, k=5, base=2)  # bits instead of nats

# The plug-in ("inv") estimator is still available directly
I_xy_plugin = mutual_information(x, y, method="inv")
```

### Conditional Entropy and Mutual Information

```julia
n = 1000
x = rand(n)
y = rand(n)
z = x + y + 0.1*rand(n)  # z depends on both x and y

# Conditional entropy: H(Z|X)  -- conditional_entropy(X, Y) = H(Y|X)
H_z_given_x = conditional_entropy(x, z)

# Conditional mutual information: I(X;Y|Z)
I_xy_given_z = conditional_mutual_information(x, y, z)
```

### Working with Matrices

For multivariate data, pass matrices:

```julia
# 1000 points in 3 dimensions
data = rand(1000, 3)

# Joint entropy H(X1, X2, X3)
H_joint = entropy(data)

# Mutual information between two multivariate variables
X = rand(1000, 2)  # 2D variable
Y = rand(1000, 2)  # 2D variable
I_XY = mutual_information(X, Y)
```

## Verifying Scale Invariance

A key feature of the invariant method is that entropy estimates don't change under scaling or translation:

```julia
x = rand(1000)

H1 = entropy(x)
H2 = entropy(1000 * x)           # Scale by 1000
H3 = entropy(x .+ 12345)         # Translate
H4 = entropy(0.001 * x .- 999)   # Both

println("Original:    $H1")
println("Scaled:      $H2")
println("Translated:  $H3")
println("Both:        $H4")
# All values should be approximately equal!
```

The same holds for mutual information -- and independently for each variable, not just
jointly:

```julia
n = 2000
x = rand(n)
y = 2*x + 0.1*rand(n)

I1 = mutual_information(x, y)
I2 = mutual_information(1e6*x .- 5, 1e-6*y .+ 3)   # very different, independent rescalings

println("Original: $I1")
println("Rescaled: $I2")
# Equal, even though x and y were rescaled by completely different factors
```

See the [Tutorial](@ref) for why this matters in practice -- a handful of extreme
outliers is where this package's normalization has a real, measurable edge over the
`std`-based normalization most other MI/CMI implementations use.

## Handling Degenerate Cases

When data has many repeated values or very small nearest-neighbor distances, the logarithm can produce strongly negative values. Use `degenerate=true` to handle this:

```julia
# Data with some repeated values
x = vcat(rand(500), fill(0.5, 500))

# May produce unexpected results
H1 = entropy(x)

# More robust for degenerate data
H2 = entropy(x, degenerate=true)
```

## Next Steps

- See the [Tutorial](@ref) for more detailed examples
- Read about the [Theory](@ref) behind the invariant method
- Browse the complete [API Reference](@ref)
