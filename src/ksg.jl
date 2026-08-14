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

# Largest radius strictly below `eps`, correct at any scale.
#
# NearestNeighbors' `inrange`/`inrangecount` use a non-strict (<=) radius comparison, but
# the KSG/Frenzel-Pompe algorithm requires strict (<) neighbor counts. Stepping down
# exactly one ULP gives that, and the step scales with the radius.
#
# Subtracting a fixed absolute epsilon instead (this used to be 1e-12, the same fix the
# `ennemi` Python package applies, see https://github.com/polsys/ennemi/issues/76) also
# drops any genuine neighbor lying within that epsilon of the shared radius. Those
# neighbors belong inside the ball, so the marginal counts come out wrong.
#
# Invariant normalization keeps *typical* distances near 1, which is what made the
# absolute epsilon look safe, but it cannot keep individual neighbors away from the
# radius. Data mixing two very different scales (a cluster orders of magnitude tighter
# than the median spacing, alongside a normal spread) puts many neighbors inside that
# window at once. The sign of the resulting error depends on which subspace loses more
# counts: a small tight spike biases MI upward, a dominant tight core biases it downward
# by as much as 0.1 nat where the truth is 0.
#
# `eps == 0` (k+1 coincident points) is handled by `_check_no_degenerate_radius` below,
# which must run BEFORE any query. This is the one place the Julia and Python versions
# genuinely differ: SciPy's `query_ball_point` accepts a negative radius and returns a
# count of 0, so the Python side can pass one through and let `_check_no_degenerate_counts`
# report it. NearestNeighbors rejects it outright ("the query radius r must be ≧ 0"), so
# here the degenerate radius has to be caught up front or the user gets that message
# instead of one describing their data.
_strict_radius(eps::Real)::Float64 = prevfloat(Float64(eps))

# Raise the same error `_check_no_degenerate_counts` would, but from the shared radius
# itself, before it reaches a query that would throw a less informative message.
function _check_no_degenerate_radius(eps::Vector{Float64})
    n_degenerate = count(==(0), eps)
    if n_degenerate > 0
        throw(ArgumentError(
            "Shared KSG radius is degenerate for $n_degenerate point(s): " *
            "at least k+1 points coincide exactly in the joint space (e.g. " *
            "multiple all-zero/duplicate rows). Cannot compute a finite " *
            "entropy estimate for these points -- consider deduplicating, " *
            "adding jitter, or excluding the offending dimension(s)."
        ))
    end
end

function _invariant_normalize_row(mat::Matrix{<:Real})::Matrix{Float64}
    measure = compute_invariant_measure(mat[1, :])
    return Matrix{Float64}(mat) ./ measure
end

# Raise a clear error if any marginal/subspace neighbor count is 0.
#
# A count of 0 means the shared KSG radius was degenerate (exactly 0) at that
# point -- i.e. at least k+1 points are exact duplicates in the joint space,
# most often because several dimensions are simultaneously sparse (e.g. many
# rows are all zero). digamma(0) is -Inf, so this would otherwise propagate
# silently into NaN.
function _check_no_degenerate_counts(counts_by_name::Pair{String,<:Vector{<:Integer}}...)
    for (name, counts) in counts_by_name
        n_degenerate = count(==(0), counts)
        if n_degenerate > 0
            throw(ArgumentError(
                "Shared KSG radius is degenerate for $n_degenerate point(s) " *
                "in the '$name' subspace: at least k+1 points coincide " *
                "exactly in the joint space (e.g. multiple all-zero/duplicate " *
                "rows). Cannot compute a finite entropy estimate for these " *
                "points -- consider deduplicating, adding jitter, or " *
                "excluding the offending dimension(s)."
            ))
        end
    end
end

# H(Xi) in nats from an already invariant-normalized 1xn row.
#
# Used for I(Xi; Xi) = H(Xi): pairing a variable with itself is never run
# through the shared-radius KSG trick below, since any duplicate value in Xi
# (e.g. repeated zeros in sparse data) then collides with itself in the
# joint (Xi, Xi) space, making the shared radius degenerate far more easily
# than a genuine two-variable pair would. The plain k-NN entropy estimate
# here tolerates duplicates by dropping degenerate (zero-distance) points
# from the log-distance average -- the same behavior as method="inv" --
# instead of hard-failing.
function _entropy_nats_from_normalized(col::Matrix{Float64}, k::Int)::Float64
    n = size(col, 2)
    tree = KDTree(col, Chebyshev())
    _, dists = knn(tree, col, k + 1, true)
    kth_dists = [d[k + 1] for d in dists]
    log_dists = log.(filter(!=(0), kth_dists))
    return compute_knn_entropy_nats(log_dists, 1, k, n)
end

# KSG MI in nats, given `x`, `y` already invariant-normalized (each a 1×n
# matrix, canonical format: one row, n columns), plus their two PRE-BUILT
# marginal (1D) KDTrees.
#
# Splitting out the marginal trees lets a caller computing many pairs over
# the same set of dimensions (the `MI` matrix function in optimized.jl) build
# each dimension's 1D tree once and reuse it across every pair it appears in,
# instead of rebuilding it from scratch for every single pair. Only the
# joint (2D) tree below is genuinely pair-specific.
function _mi_ksg_pair(x::Matrix{Float64}, y::Matrix{Float64}, x_tree, y_tree, k::Int)::Float64
    n = size(x, 2)
    xy = vcat(x, y)

    joint_tree = KDTree(xy, Chebyshev())

    # Shared radius: k-th neighbor distance in the joint (normalized) space.
    _, dists = knn(joint_tree, xy, k + 1, true)
    eps = [d[k + 1] for d in dists]
    _check_no_degenerate_radius(eps)

    nx = [inrangecount(x_tree, x[:, i], _strict_radius(eps[i])) for i in 1:n]
    ny = [inrangecount(y_tree, y[:, i], _strict_radius(eps[i])) for i in 1:n]
    _check_no_degenerate_counts("x" => nx, "y" => ny)

    return digamma(n) + digamma(k) - mean(digamma.(nx) .+ digamma.(ny))
end

# KSG MI in nats, given `x`, `y` already invariant-normalized, each a 1×n matrix
# (canonical format: one row, n columns).
function _mi_ksg_from_normalized(x::Matrix{Float64}, y::Matrix{Float64}, k::Int)::Float64
    return _mi_ksg_pair(x, y, KDTree(x, Chebyshev()), KDTree(y, Chebyshev()), k)
end

# Frenzel-Pompe CMI in nats, given `x`, `y`, `z` already invariant-normalized
# (each a 1×n matrix), plus PRE-BUILT (X,Z) and (Y,Z) subspace trees and the
# Z tree.
#
# Splitting these out lets a caller computing many pairs against the same
# conditioning variable Z (the `CMI` matrix function in optimized.jl) build
# each dimension's (Xi, Z) tree once and the single Z tree once, and reuse
# them across every pair -- instead of rebuilding all three (plus Z, which
# never changes) from scratch for every single pair. Only the full joint
# (3D) tree below is genuinely pair-specific.
function _cmi_fp_pair(x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64}, xz_tree, yz_tree, z_tree, k::Int)::Float64
    n = size(x, 2)
    xyz = vcat(x, y, z)
    xz = vcat(x, z)
    yz = vcat(y, z)

    full_tree = KDTree(xyz, Chebyshev())

    # Shared radius: k-th neighbor distance in the full joint (normalized) space.
    _, dists = knn(full_tree, xyz, k + 1, true)
    eps = [d[k + 1] for d in dists]
    _check_no_degenerate_radius(eps)

    nxz = [inrangecount(xz_tree, xz[:, i], _strict_radius(eps[i])) for i in 1:n]
    nyz = [inrangecount(yz_tree, yz[:, i], _strict_radius(eps[i])) for i in 1:n]
    nz  = [inrangecount(z_tree,  z[:, i],  _strict_radius(eps[i])) for i in 1:n]
    _check_no_degenerate_counts("x,z" => nxz, "y,z" => nyz, "z" => nz)

    return digamma(k) - mean(digamma.(nxz) .+ digamma.(nyz) .- digamma.(nz))
end

# Frenzel-Pompe CMI in nats, given `x`, `y`, `z` already invariant-normalized, each a
# 1×n matrix (canonical format: one row, n columns).
function _cmi_fp_from_normalized(x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64}, k::Int)::Float64
    xz_tree = KDTree(vcat(x, z), Chebyshev())
    yz_tree = KDTree(vcat(y, z), Chebyshev())
    z_tree = KDTree(z, Chebyshev())
    return _cmi_fp_pair(x, y, z, xz_tree, yz_tree, z_tree, k)
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

    if mat_1_canonical == mat_2_canonical
        # I(X;X) = H(X) exactly. Skip the shared-radius trick: pairing X with
        # itself makes any duplicate value in X collide with itself in the
        # joint (X, X) space, so it hits the degenerate-radius case far more
        # easily than a genuine two-variable pair -- see
        # _entropy_nats_from_normalized above.
        mi_nats = _entropy_nats_from_normalized(x, k)
        return convert_to_base(mi_nats, base)
    end

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
