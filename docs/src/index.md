# EntropyInvariant.jl

*An improved nearest neighbor method for estimating differential entropy*

## Overview

EntropyInvariant.jl is a Julia package implementing an improved k-nearest neighbor method for estimating differential entropy for continuous variables. The key innovation is an **invariant measure** $m(x)$ based on the median value of nearest-neighbor distances, solving Edwin Thompson Jaynes' limiting density of discrete points problem.

```math
H(X) = -\int_X p(x)\log\left(\frac{p(x)}{m(x)}\right)\mathrm{d}x
```

## Key Features

- **Invariant under change of variables**: Scaling and translation do not affect entropy estimates
- **Always positive**: Unlike standard k-NN methods that can produce negative values
- **Multiple estimation methods**: Invariant (default), k-NN, and histogram-based
- **Comprehensive information theory**: Entropy, mutual information, conditional entropy, and more
- **Partial Information Decomposition**: Redundancy, unique information, and synergy

## Quick Example

```julia
using EntropyInvariant

# Generate sample data
n = 1000
x = rand(n)
y = 2*x + 0.1*rand(n)  # y is correlated with x

# Compute entropy
H_x = entropy(x)
println("H(X) = $H_x")

# Compute mutual information
I_xy = mutual_information(x, y)
println("I(X;Y) = $I_xy")

# Verify scale invariance
H_scaled = entropy(1000 * x .+ 42)  # Scale and shift
println("H(1000X + 42) = $H_scaled")  # Same as H(X)!
```

## The Invariant Measure

The invariant measure $r_X$ satisfies three key properties:

```math
\begin{aligned}
1) & \quad m(X) = r_X > 0 \\
2) & \quad m(aX) = am(X) = ar_X \\
3) & \quad m(X+b) = m(X) = r_X
\end{aligned}
```

We found that the **median value of nearest-neighbor distances** multiplied by the number of points provides an appropriate measure satisfying these properties.

## Available Functions

### Basic Quantities
- [`entropy`](@ref EntropyInvariant.entropy) - Differential entropy $H(X)$
- `conditional_entropy` - Conditional entropy $H(X|Y)$
- [`mutual_information`](@ref EntropyInvariant.mutual_information) - Mutual information $I(X;Y)$

### Advanced Quantities
- `conditional_mutual_information` - $I(X;Y|Z)$
- [`normalized_mutual_information`](@ref EntropyInvariant.normalized_mutual_information) - NMI normalized to $[0,1]$
- [`interaction_information`](@ref EntropyInvariant.interaction_information) - Three-way interaction
- `information_quality_ratio` - Ratio of mutual to marginal information

### Partial Information Decomposition
- [`redundancy`](@ref EntropyInvariant.redundancy) - Shared information $R(X,Y;Z)$
- [`unique`](@ref EntropyInvariant.unique) - Unique information each variable provides
- `synergy` - Combined information beyond individual contributions

### Optimized Matrix Functions
- [`MI`](@ref EntropyInvariant.MI) - Pairwise mutual information matrix
- [`CMI`](@ref EntropyInvariant.CMI) - Pairwise conditional MI matrix

## Installation

```julia
using Pkg
Pkg.add("EntropyInvariant")
```

## Contents

```@contents
Pages = ["getting_started.md", "tutorial.md", "theory.md", "api.md"]
Depth = 2
```

## Authors

- Félix Truong (SOLEIL Synchrotron)
- Alexandre Giuliani (SOLEIL Synchrotron)

## References

1. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). [Estimating mutual information](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138). Physical Review E, 69(6), 066138.
2. Jaynes, E. T. (1968). [Prior Probabilities](https://ieeexplore.ieee.org/document/4082152). IEEE Transactions on Systems Science and Cybernetics, 4(3), 227-241.
