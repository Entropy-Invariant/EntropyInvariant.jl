# Computation helper functions for EntropyInvariant package

"""
    compute_invariant_measure(data::Vector{<:Real}) -> Real

Compute the invariant measure r_X for a 1D dataset.

This is the core innovation of the invariant entropy method, solving Edwin Thompson
Jaynes' limiting density of discrete points problem. The invariant measure makes the
entropy estimate invariant under scaling and translation transformations.

# Algorithm
1. Filter out zero values (to handle sparse data)
2. Sort the non-zero data
3. Compute nearest-neighbor distances using `nn1()`
4. Take median of nearest-neighbor distances
5. Multiply by number of points to get scale-invariant measure

Formula: r_X = median(nearest_neighbor_distances) × num_points

# Arguments
- `data::Vector{<:Real}`: 1D data vector

# Returns
- `Real`: The invariant measure r_X

# Example
```julia
x = rand(1000)
r_x = compute_invariant_measure(x)
# r_x represents the characteristic scale of the data distribution
```
"""
function compute_invariant_measure(data::Vector{<:Real})::Real
    non_zero_data = filter(x -> x != 0, data)
    sorted_data = sort(non_zero_data)
    nn_distances = nn1(sorted_data)
    median_distance = median(nn_distances)
    num_points = length(non_zero_data)
    return median_distance * num_points
end

"""
    normalize_by_invariant_measure(mat::Matrix{<:Real}) -> Matrix{Float64}

Normalize each dimension of a matrix by its invariant measure.

This transformation ensures that the entropy estimate is invariant under
change of variables (scaling and translation). Each row (dimension) is
independently normalized by its own characteristic scale.

Eliminates 4 instances of complex inline invariant measure computation.

# Arguments
- `mat::Matrix{<:Real}`: Data matrix in canonical format (rows=dimensions, cols=points)

# Returns
- `Matrix{Float64}`: Normalized matrix where mat[i,:] /= r_X[i] for each dimension i

# Example
```julia
mat = rand(3, 1000)  # 3 dimensions, 1000 points
normalized = normalize_by_invariant_measure(mat)
# Each dimension now has unit invariant measure
```
"""
function normalize_by_invariant_measure(mat::Matrix{<:Real})::Matrix{Float64}
    normalized = Matrix{Float64}(copy(mat))
    num_dims = size(mat, 1)

    for i in 1:num_dims
        measure = compute_invariant_measure(mat[i, :])
        normalized[i, :] ./= measure
    end

    return normalized
end

"""
    compute_knn_distances(mat::Matrix{<:Real}, k::Int) -> KNNResult

Compute k-nearest neighbor distances for all points in a dataset.

Uses KDTree for efficient nearest neighbor search. Eliminates 8 instances
of duplicate KDTree creation and distance extraction throughout the codebase.

# Arguments
- `mat::Matrix{<:Real}`: Data matrix in canonical format (columns as points)
- `k::Int`: Number of nearest neighbors (excluding the point itself)

# Returns
- `KNNResult`: Struct containing indices, distances, and k-th distances

# Algorithm
1. Build KDTree for efficient spatial search
2. Query k+1 neighbors for each point (including the point itself)
3. Extract distance to k-th neighbor (index k+1, since 1st is the point itself)

# Example
```julia
mat = rand(3, 1000)  # 3 dimensions, 1000 points
knn = compute_knn_distances(mat, 3)
# knn.kth_distances[i] is distance from point i to its 3rd nearest neighbor
```
"""
function compute_knn_distances(mat::Matrix{<:Real}, k::Int)::KNNResult
    kdtree = KDTree(mat)
    indices, distances = knn(kdtree, mat, k + 1, true)
    kth_distances = [dists[k + 1] for dists in distances]
    return KNNResult(indices, distances, kth_distances)
end

"""
    extract_nonzero_log_distances(distances::Vector{Float64}, noise::Int) -> Vector{Float64}

Extract logarithms of non-zero distances, optionally adding noise for degenerate cases.

Filters out zero distances (which would cause log(0) = -∞) and optionally adds
1 to handle degenerate cases where distances are very small.

# Arguments
- `distances::Vector{Float64}`: Vector of distances
- `noise::Int`: 0 for normal mode, 1 for degenerate mode (adds 1 before log)

# Returns
- `Vector{Float64}`: Log of non-zero distances (with optional +1)

# Example
```julia
dists = [0.1, 0.0, 0.5, 0.2]
log_dists = extract_nonzero_log_distances(dists, 0)
# Returns: [log(0.1), log(0.5), log(0.2)] (skips 0.0)

log_dists_deg = extract_nonzero_log_distances(dists, 1)
# Returns: [log(1.1), log(1.5), log(1.2)] (adds 1, skips 0.0)
```
"""
function extract_nonzero_log_distances(distances::Vector{Float64}, noise::Int)::Vector{Float64}
    return [log(d + noise) for d in distances if d != 0]
end

"""
Unit ball volumes for dimensions 1-3, used in k-NN entropy estimation.

- V₁ = 2.0 (line segment: [-1, 1])
- V₂ = π ≈ 3.14159 (circle with radius 1)
- V₃ = 4π/3 ≈ 4.18879 (sphere with radius 1)

For dimensions > 3, the volume can be computed using:
V_d = π^(d/2) / Γ(d/2 + 1)

These constants eliminate magic numbers throughout the codebase.
"""
const UNIT_BALL_VOLUMES = [2.0, π, 4π/3]

"""
Precomputed logarithms of unit ball volumes for computational efficiency.
"""
const LOG_UNIT_BALL_VOLUMES = log.(UNIT_BALL_VOLUMES)

"""
    compute_knn_entropy_nats(log_distances::Vector{Float64}, dimension::Int, k::Int) -> Float64

Compute k-NN entropy estimate in nats (natural logarithm base).

Implements the Kraskov-Stögbauer-Grassberger (2004) k-NN entropy estimator:

    H = d⋅mean(log(ρₖ)) + log(V_d) + ψ(n) - ψ(k)

where:
- d: dimension of the space
- ρₖ: distance to k-th nearest neighbor for each point
- V_d: volume of d-dimensional unit ball
- ψ: digamma function
- n: number of points
- k: number of neighbors

# Arguments
- `log_distances::Vector{Float64}`: Logarithms of k-th nearest neighbor distances
- `dimension::Int`: Dimensionality of the space (1, 2, or 3)
- `k::Int`: Number of neighbors used

# Returns
- `Float64`: Entropy estimate in nats (base e)

# Reference
Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
Estimating mutual information. Physical Review E, 69(6), 066138.

# Example
```julia
log_dists = [log(0.1), log(0.2), log(0.15), ...]  # n distances
H = compute_knn_entropy_nats(log_dists, 2, 3)  # 2D space, k=3
```
"""
function compute_knn_entropy_nats(
    log_distances::Vector{Float64},
    dimension::Int,
    k::Int
)::Float64
    if dimension < 1 || dimension > 3
        throw(ArgumentError("Unit ball volume only available for dimensions 1-3, got $dimension"))
    end

    n = length(log_distances)
    mean_log_dist = mean(log_distances)
    log_volume = LOG_UNIT_BALL_VOLUMES[dimension]

    entropy = (dimension * mean_log_dist +
               log_volume +
               digamma(n) -
               digamma(k))

    return entropy
end

"""
    convert_to_base(entropy_nats::Float64, base::Real) -> Float64

Convert entropy from nats (base e) to specified logarithmic base.

Conversion formula: H_base = H_nats × log_e(base)

# Arguments
- `entropy_nats::Float64`: Entropy in nats (natural logarithm)
- `base::Real`: Target logarithmic base (e.g., 2 for bits, 10 for dits, e for nats)

# Returns
- `Float64`: Entropy in specified base

# Example
```julia
H_nats = 2.5
H_bits = convert_to_base(H_nats, 2)  # Convert to bits
H_dits = convert_to_base(H_nats, 10)  # Convert to decimal digits
```
"""
function convert_to_base(entropy_nats::Float64, base::Real)::Float64
    return entropy_nats * log(base, e)
end
