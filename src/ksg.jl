# KSG / Frenzel-Pompe estimators, computed after invariant-measure normalization.
#
# The plug-in estimators elsewhere in this package build MI and CMI by differencing
# independent Kozachenko-Leonenko entropy estimates (H(X) + H(Y) - H(X,Y), etc). Each
# term picks its own k-NN radius independently, so their finite-sample biases don't
# cancel -- this shows up as systematic drift on outlier-contaminated or near-degenerate
# (low-rank manifold) data.
#
# KSG (Kraskov, Stogbauer & Grassberger, 2004) and its conditional extension, Frenzel &
# Pompe (2007), fix this for the *unnormalized* case by finding the k-th neighbor radius
# once in the full joint space (Chebyshev metric) and reusing that same radius to count
# neighbors in each marginal/subspace -- the shared radius makes the leading-order bias
# terms cancel algebraically.
#
# This module combines both ideas: each variable is first normalized by its own
# invariant measure (median nearest-neighbor distance, `compute_invariant_measure`) --
# giving affine scale-invariance and outlier-robustness the same way `entropy_inv` does
# -- and then KSG/Frenzel-Pompe's shared-radius neighbor counting is applied on the
# normalized data, giving the bias cancellation that the plug-in formulas lack.
#
# The `_from_normalized` functions are the reusable core (used directly by the `MI`/`CMI`
# matrix functions in optimized.jl, which normalize each column once up front rather than
# repeating it per pair).

# NearestNeighbors' `inrange`/`inrangecount` use a non-strict (<=) radius comparison, but
# the KSG/Frenzel-Pompe algorithm requires strict (<) neighbor counts. Subtracting a small
# epsilon from the radius corrects this without materially affecting real (roughly
# unit-magnitude, post-normalization) distances. Same fix used by the `ennemi` Python
# package (see https://github.com/polsys/ennemi/issues/76).
const _STRICT_RADIUS_EPS = 1e-12

function _invariant_normalize_row(mat::Matrix{<:Real})::Matrix{Float64}
    measure = compute_invariant_measure(mat[1, :])
    return Matrix{Float64}(mat) ./ measure
end

# KSG MI in nats, given `x`, `y` already invariant-normalized, each a 1×n matrix
# (canonical format: one row, n columns).
function _mi_ksg_from_normalized(x::Matrix{Float64}, y::Matrix{Float64}, k::Int)::Float64
    n = size(x, 2)
    xy = vcat(x, y)

    joint_tree = KDTree(xy, Chebyshev())
    x_tree = KDTree(x, Chebyshev())
    y_tree = KDTree(y, Chebyshev())

    # Shared radius: k-th neighbor distance in the joint (normalized) space.
    _, dists = knn(joint_tree, xy, k + 1, true)
    eps = [d[k + 1] for d in dists]

    nx = [inrangecount(x_tree, x[:, i], eps[i] - _STRICT_RADIUS_EPS) for i in 1:n]
    ny = [inrangecount(y_tree, y[:, i], eps[i] - _STRICT_RADIUS_EPS) for i in 1:n]

    return digamma(n) + digamma(k) - mean(digamma.(nx) .+ digamma.(ny))
end

# Frenzel-Pompe CMI in nats, given `x`, `y`, `z` already invariant-normalized, each a
# 1×n matrix (canonical format: one row, n columns).
function _cmi_fp_from_normalized(x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64}, k::Int)::Float64
    n = size(x, 2)
    xyz = vcat(x, y, z)
    xz = vcat(x, z)
    yz = vcat(y, z)

    full_tree = KDTree(xyz, Chebyshev())
    xz_tree = KDTree(xz, Chebyshev())
    yz_tree = KDTree(yz, Chebyshev())
    z_tree = KDTree(z, Chebyshev())

    # Shared radius: k-th neighbor distance in the full joint (normalized) space.
    _, dists = knn(full_tree, xyz, k + 1, true)
    eps = [d[k + 1] for d in dists]

    nxz = [inrangecount(xz_tree, xz[:, i], eps[i] - _STRICT_RADIUS_EPS) for i in 1:n]
    nyz = [inrangecount(yz_tree, yz[:, i], eps[i] - _STRICT_RADIUS_EPS) for i in 1:n]
    nz  = [inrangecount(z_tree,  z[:, i],  eps[i] - _STRICT_RADIUS_EPS) for i in 1:n]

    return digamma(k) - mean(digamma.(nxz) .+ digamma.(nyz) .- digamma.(nz))
end

"""
    mutual_information_ksg(X, Y; k::Int = 3, base::Real = e, verbose::Bool = false, dim::Int = 1) -> Real

KSG mutual information estimator, applied after invariant-measure normalization.

`X` and `Y` must each be 1-dimensional (a `Vector`, or a `Matrix` with exactly one row
or one column depending on `dim`).

# Arguments
- `X`, `Y`: First and second variable (1-dimensional).
- `k::Int = 3`: Number of neighbors.
- `base::Real = e`: Logarithmic base.
- `verbose::Bool = false`: Print computation info.
- `dim::Int = 1`: Data layout (1: points as rows, 2: points as columns).

# Returns
- `Real`: Mutual information I(X;Y).
"""
function mutual_information_ksg(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, dim::Int = 1)::Real
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)

    validate_same_num_points([shape1, shape2])
    validate_dimensions_equal_one([shape1, shape2])
    verbose && log_computation_info([shape1, shape2], base)

    x = _invariant_normalize_row(mat_1_canonical)
    y = _invariant_normalize_row(mat_2_canonical)

    mi_nats = _mi_ksg_from_normalized(x, y, k)
    return convert_to_base(mi_nats, base)
end

function mutual_information_ksg(array_1::Vector{<:Real}, array_2::Vector{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false)::Real
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    return mutual_information_ksg(mat_1, mat_2, k=k, base=base, verbose=verbose)
end

"""
    conditional_mutual_information_ksg(X, Y, Z; k::Int = 3, base::Real = e, verbose::Bool = false, dim::Int = 1) -> Real

Frenzel-Pompe conditional mutual information estimator, applied after invariant-measure
normalization.

`X`, `Y`, `Z` must each be 1-dimensional.

# Arguments
- `X`, `Y`: First and second variable (1-dimensional).
- `Z`: Conditioning variable (1-dimensional).
- `k::Int = 3`: Number of neighbors.
- `base::Real = e`: Logarithmic base.
- `verbose::Bool = false`: Print computation info.
- `dim::Int = 1`: Data layout (1: points as rows, 2: points as columns).

# Returns
- `Real`: Conditional mutual information I(X;Y|Z).
"""
function conditional_mutual_information_ksg(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, cond_::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, dim::Int = 1)::Real
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    cond_canonical = ensure_columns_are_points(cond_, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)
    shape3 = get_shape(cond_canonical)

    validate_same_num_points([shape1, shape2, shape3])
    validate_dimensions_equal_one([shape1, shape2, shape3])
    verbose && log_computation_info([shape1, shape2, shape3], base)

    x = _invariant_normalize_row(mat_1_canonical)
    y = _invariant_normalize_row(mat_2_canonical)
    z = _invariant_normalize_row(cond_canonical)

    cmi_nats = _cmi_fp_from_normalized(x, y, z, k)
    return convert_to_base(cmi_nats, base)
end

function conditional_mutual_information_ksg(array_1::Vector{<:Real}, array_2::Vector{<:Real}, cond_::Vector{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false)::Real
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    cond_mat = vector_to_matrix(cond_)
    return conditional_mutual_information_ksg(mat_1, mat_2, cond_mat, k=k, base=base, verbose=verbose)
end
