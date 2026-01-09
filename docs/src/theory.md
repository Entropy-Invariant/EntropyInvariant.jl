# Theory

This section describes the mathematical foundations of the invariant entropy estimation method.

## The Problem with Differential Entropy

Shannon entropy for discrete random variables is well-defined:

```math
H(X) = -\sum_i p_i \log p_i
```

However, extending this to continuous variables presents challenges. The naive differential entropy:

```math
H(X) = -\int p(x) \log p(x) \, dx
```

has several issues:

1. **Not invariant under change of variables**: If $Y = g(X)$, then generally $H(Y) \neq H(X)$
2. **Can be negative**: Unlike discrete entropy
3. **Depends on units**: Scaling the variable changes the entropy

## Jaynes' Limiting Density of Discrete Points

Edwin Thompson Jaynes proposed that the proper generalization of entropy to continuous variables should include an *invariant measure* $m(x)$:

```math
H(X) = -\int p(x) \log \frac{p(x)}{m(x)} \, dx
```

This formulation:
- Reduces to Shannon entropy in the discrete limit
- Is invariant under coordinate transformations (if $m(x)$ transforms appropriately)
- Represents information relative to a reference measure

The challenge is: what is the appropriate $m(x)$?

## The Invariant Measure

We propose an invariant measure $r_X$ based on nearest-neighbor distances that satisfies:

```math
\begin{aligned}
1) & \quad m(X) = r_X > 0 && \text{(positivity)} \\
2) & \quad m(aX) = a \cdot m(X) = a \cdot r_X && \text{(scale covariance)} \\
3) & \quad m(X+b) = m(X) = r_X && \text{(translation invariance)}
\end{aligned}
```

### Construction

Given $n$ samples $\{x_1, \ldots, x_n\}$:

1. **Sort the data** for each dimension
2. **Compute nearest-neighbor distances**: For each point, find the distance to its nearest neighbor
3. **Take the median**: $\tilde{d} = \text{median}(\{d_1, \ldots, d_n\})$
4. **Scale by sample size**: $r_X = \tilde{d} \cdot n$

The use of the median (rather than mean) provides robustness to outliers.

## k-NN Entropy Estimation

The standard k-NN entropy estimator (Kraskov et al., 2004) is:

```math
\hat{H}(X) = -\psi(k) + \psi(n) + \log V_d + \frac{d}{n} \sum_{i=1}^n \log \rho_k(i)
```

where:
- $\psi$ is the digamma function
- $n$ is the sample size
- $k$ is the number of nearest neighbors
- $d$ is the dimension
- $V_d$ is the volume of the unit ball in $d$ dimensions
- $\rho_k(i)$ is the distance from point $i$ to its $k$-th nearest neighbor

### Unit Ball Volumes

```math
V_d = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)}
```

For common dimensions:
- $V_1 = 2$
- $V_2 = \pi$
- $V_3 = \frac{4\pi}{3}$

## Invariant k-NN Estimator

Our invariant estimator modifies the standard k-NN approach:

1. **Normalize each dimension** by its invariant measure:
   ```math
   \tilde{x}_j = \frac{x_j}{r_{X_j}}
   ```

2. **Apply standard k-NN estimation** to the normalized data

This ensures that the entropy estimate is invariant under scaling and translation of the original variables.

## Information-Theoretic Quantities

All other quantities are derived from entropy using standard identities:

### Conditional Entropy

```math
H(X|Y) = H(X,Y) - H(Y)
```

### Mutual Information

```math
I(X;Y) = H(X) + H(Y) - H(X,Y)
```

Equivalent forms:
```math
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```

### Conditional Mutual Information

```math
I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
```

### Interaction Information

```math
I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
```

This can be positive (redundancy dominates) or negative (synergy dominates).

### Normalized Mutual Information

```math
\text{NMI}(X;Y) = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)}
```

Bounded in $[0, 1]$ with 1 indicating perfect dependence.

## Partial Information Decomposition

PID decomposes the mutual information $I(\{X,Y\};Z)$ into:

- **Redundancy** $R(X,Y;Z)$: Information that both X and Y provide about Z
- **Unique information** $U(X;Z)$: Information only X provides
- **Unique information** $U(Y;Z)$: Information only Y provides
- **Synergy** $S(X,Y;Z)$: Information only available from X and Y together

```math
I(X,Y;Z) = R(X,Y;Z) + U(X;Z) + U(Y;Z) + S(X,Y;Z)
```

### Minimum Mutual Information (MMI) Approach

We use the MMI approach for redundancy:

```math
R(X,Y;Z) = \min(I(X;Z), I(Y;Z))
```

Then:
```math
\begin{aligned}
U(X;Z) &= I(X;Z) - R(X,Y;Z) \\
U(Y;Z) &= I(Y;Z) - R(X,Y;Z) \\
S(X,Y;Z) &= I(X,Y;Z) - I(X;Z) - I(Y;Z) + R(X,Y;Z)
\end{aligned}
```

## Comparison of Methods

| Method | Pros | Cons |
|--------|------|------|
| **Invariant (inv)** | Scale/translation invariant, positive | Slightly higher computational cost |
| **k-NN (knn)** | Well-studied, efficient | Can be negative, not invariant |
| **Histogram** | Simple, fast | Requires binning, curse of dimensionality |

## References

1. **Kraskov, A., Stögbauer, H., & Grassberger, P.** (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

2. **Jaynes, E. T.** (1968). Prior Probabilities. *IEEE Transactions on Systems Science and Cybernetics*, 4(3), 227-241.

3. **Kozachenko, L. F., & Leonenko, N. N.** (1987). Sample estimate of the entropy of a random vector. *Problemy Peredachi Informatsii*, 23(2), 9-16.

4. **Williams, P. L., & Beer, R. D.** (2010). Nonnegative decomposition of multivariate information. *arXiv preprint arXiv:1004.2515*.
