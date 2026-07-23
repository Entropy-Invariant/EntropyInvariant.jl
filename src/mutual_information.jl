# Mutual Information Computation Functions
#
# This file contains implementations for computing mutual information and conditional entropy
# between datasets. It includes:
# - conditional_entropy: Compute H(X|Y) = H(X,Y) - H(X)
# - mutual_information: Compute I(X;Y) = H(X) + H(Y) - H(X,Y)
# - conditional_mutual_information: Compute I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)

"""
    conditional_entropy(X, Y; method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute conditional entropy H(Y|X) = H(X,Y) - H(X).

`method = "inv_ksg"` (default) computes H(Y) - I(X;Y) using the plain invariant entropy
for H(Y) and the bias-cancelling KSG estimator (see `mutual_information`) for I(X;Y).
`"inv"`, `"knn"`, and `"histogram"` instead use the plug-in formula H(X,Y) - H(X),
differencing two independently-estimated entropies.
"""
function conditional_entropy(mat_::Matrix{<:Real}, cond_::Matrix{<:Real};method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    if method == "inv_ksg"
        ent_cond = entropy(cond_, method="inv", k=k, base=base, dim=dim)
        mi_val = mutual_information_ksg(mat_, cond_, k=k, base=base, verbose=verbose, dim=dim)
        return ent_cond - mi_val
    end

    # Preprocessing: normalize data layout and extract shapes
    mat = ensure_columns_are_points(mat_, dim)
    cond = ensure_columns_are_points(cond_, dim)
    shape_mat = get_shape(mat)
    shape_cond = get_shape(cond)

    # Validation
    validate_same_num_points([shape_mat, shape_cond])
    validate_dimensions_equal_one([shape_mat, shape_cond])
    verbose && log_computation_info([shape_mat, shape_cond], base)

    # Compute conditional entropy: H(X,Y) - H(X)
    # Note: Returns H(Y|X) where mat_ is X and cond_ is Y
    joint_mat = vcat(mat, cond)
    ent_mat = entropy(mat, k=k, method=method, nbins=nbins, base=base, degenerate=degenerate, dim=2)
    ent_joint = entropy(joint_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)

    return ent_joint - ent_mat
end

function conditional_entropy(array_::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    array_mat = vector_to_matrix(array_)
    cond_mat = vector_to_matrix(cond_)
    return conditional_entropy(array_mat, cond_mat, method=method, k=k, nbins=nbins, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    mutual_information(X::Matrix{<:Real}, Y::Union{Matrix{<:Real}, Nothing} = nothing; method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1, optimize::Bool = false) -> Real

Compute the mutual information between two datasets as I(X; Y) = H(X) + H(Y) - H(X, Y)
  where:
  - H(X): Entropy of the first dataset X.
  - H(Y): Entropy of the second dataset Y.
  - H(X, Y): Joint entropy of X and Y.

# Arguments
- X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset.
- Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset.
- `method::String = "inv_ksg"` (optional): The method to use for MI computation. Options are:
  - `"inv_ksg"` (default): Normalizes X and Y by their invariant measure and applies the
    KSG (Kraskov et al. 2004) shared-radius estimator directly, which cancels the
    leading-order k-NN bias that the plug-in formula below does not -- this matters most
    on outlier-contaminated or near-degenerate data.
  - `"inv"`, `"knn"`, `"histogram"`: Build MI from independently-estimated H(X), H(Y),
    H(X,Y) (plug-in formula). Use these if you need the individual entropy terms, not
    just I(X;Y).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for `"inv_ksg"` and the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).
- `optimize::Bool = false` (optional): If `true`, compute the two-dimensional mutual_information of X faster (via `MI`). Defaults to `false`. Y should be nothing.

# Returns
- `Real`: The computed mutual information I(X; Y) between the datasets X and Y.

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called, or the `mutual_information_ksg` estimator directly.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D

# Using the default (invariant measure + KSG)
mi = mutual_information(x, y)

# Using k-NN method
mi = mutual_information(x, y, method="knn", k=5, verbose=true)

# Using histogram method
mi = mutual_information(x, y, method="histogram", nbins=10)

# Using invariant plug-in method
mi = mutual_information(x, y, method="inv", k=3)
"""
function mutual_information(mat_1::Matrix{<:Real}, mat_2::Union{Matrix{<:Real}, Nothing} = nothing;method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1, optimize::Bool = false)::Real
    # Use optimized matrix computation if requested
    if optimize
        return MI(mat_1, method=method, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim)
    end

    if method == "inv_ksg"
        return mutual_information_ksg(mat_1, mat_2, k=k, base=base, verbose=verbose, dim=dim)
    end

    # Preprocessing: normalize data layout and extract shapes
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)

    # Validation
    validate_same_num_points([shape1, shape2])
    validate_dimensions_equal_one([shape1, shape2])
    verbose && log_computation_info([shape1, shape2], base)

    # Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    joint_mat = vcat(mat_1_canonical, mat_2_canonical)
    ent_1 = entropy(mat_1_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_joint = entropy(joint_mat, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)

    return ent_1 + ent_2 - ent_joint
end

function mutual_information(array_1::Vector{<:Real}, array_2::Union{Vector{<:Real}, Nothing} = nothing;method::String = "inv_ksg", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, optimize::Bool = false)::Real
    # Use optimized computation if requested
    if optimize
        return MI(array_1, method=method, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=1)
    end

    # Convert vectors to matrices and delegate to Matrix version
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    return mutual_information(mat_1, mat_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base, optimize=optimize)
end
                                


"""
    conditional_mutual_information(X::Matrix{<:Real}, Y::Union{Matrix{<:Real}, Nothing} = nothing, Z::Union{Matrix{<:Real}, Nothing} = nothing; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1, optimize::Bool = false) -> Real

Compute the conditional mutual information (CMI) between two datasets given a third conditioning dataset as   I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z), where:
  - H(Z): Entropy of the conditioning dataset Z.
  - H(X, Z): Joint entropy of X and Z.
  - H(Y, Z): Joint entropy of Y and Z.
  - H(X, Y, Z): Joint entropy of X, y and Z.


# Arguments
- X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the first dataset.
- Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the second dataset.
- Z::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the conditioning dataset.
- `method::String = "inv"` (optional): The method to use for entropy computation. Options are:
  - `"knn"`: k-Nearest Neighbors based entropy estimation.
  - `"histogram"`: Histogram-based entropy estimation.
  - `"inv"`: Invariant entropy estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).
- `optimize::Bool = false` (optional): If `true`, compute the invariant two-dimensional conditional mutual information of X and Z faster. Defaults to `false`. Y should be nothing.

# Returns
- `Real`: The computed conditional mutual information I(X; Y | Z)

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D
z = rand(1, 100)  # 100 points in 1D

# Using k-NN method
cmi = conditional_mutual_information(x, y, z, method="knn", k=5, verbose=true)

# Using histogram method
cmi = conditional_mutual_information(x, y, z, method="histogram", nbins=10)

# Using invariant method
cmi = conditional_mutual_information(x, y, z, method="inv", k=3)
"""
