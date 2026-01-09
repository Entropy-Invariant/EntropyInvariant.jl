# Advanced Information Theory Functions
#
# This file contains implementations for advanced information-theoretic quantities
# including conditional mutual information, normalized mutual information,
# and interaction information between datasets.
#
# Functions included:
# - conditional_mutual_information: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
# - normalized_mutual_information: NMI normalized to [0,1] range
# - interaction_information: Three-way interaction between variables
# - information_quality_ratio: Ratio of mutual to marginal information

function conditional_mutual_information(mat_1::Matrix{<:Real}, mat_2::Union{Matrix{<:Real}, Nothing} = nothing, cond_::Union{Matrix{<:Real}, Nothing} = nothing;method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1, optimize::Bool = false)::Real
    # Validate arguments
    if cond_ === nothing
        throw(ArgumentError("Conditional value is missing"))
    end

    # Use optimized matrix computation if requested
    if optimize
        return CMI(mat_1, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim)
    end

    # Preprocessing: normalize data layout and extract shapes
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    cond_canonical = ensure_columns_are_points(cond_, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)
    shape3 = get_shape(cond_canonical)

    # Validation
    validate_same_num_points([shape1, shape2, shape3])
    validate_dimensions_equal_one([shape1, shape2, shape3])
    verbose && log_computation_info([shape1, shape2, shape3], base)

    # Compute conditional mutual information: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    ent_z = entropy(cond_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_xz = entropy(vcat(mat_1_canonical, cond_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_yz = entropy(vcat(mat_2_canonical, cond_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_xyz = entropy(vcat(mat_1_canonical, mat_2_canonical, cond_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)

    return ent_xz + ent_yz - ent_xyz - ent_z
end

function conditional_mutual_information(array_1::Vector{<:Real}, array_2::Union{Vector{<:Real}, Nothing} = nothing, cond_::Union{Vector{<:Real}, Nothing} = nothing;method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, optimize::Bool = false)::Real
    # Validate arguments
    if cond_ === nothing
        throw(ArgumentError("Conditional value is missing"))
    end

    # Use optimized computation if requested
    if optimize
        return CMI(array_1, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=1)
    end

    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    cond_mat = vector_to_matrix(cond_)

    return conditional_mutual_information(mat_1, mat_2, cond_mat, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base, optimize=optimize)
end



"""
    normalized_mutual_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the normalized mutual information (NMI) between two datasets as  NMI(X, Y) = max(0, H(X) + H(Y) - H(X, Y))/((1/2)*(H(X) + H(Y))
  where:
  - H(X): Entropy of the first dataset X.
  - H(Y): Entropy of the second dataset Y`
  - H(X, Y): Joint entropy of X and Y.


# Arguments
- X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the first dataset.
- Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the second dataset.
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

# Returns
- `Real`: The normalized mutual information NMI(X, Y) between X and Y.
  - 0 < NMI < 1, where 0 indicates no shared information and 1 indicates complete mutual information.

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D

# Using k-NN method
nmi = normalized_mutual_information(x, y, method="knn", k=5, verbose=true)

# Using histogram method
nmi = normalized_mutual_information(x, y, method="histogram", nbins=10)

# Using invariant method
nmi = normalized_mutual_information(x, y, method="inv", k=3)
"""
function normalized_mutual_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    # Preprocessing: normalize data layout and extract shapes
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)

    # Validation
    validate_same_num_points([shape1, shape2])
    validate_dimensions_equal_one([shape1, shape2])
    verbose && log_computation_info([shape1, shape2], base)

    # Compute normalized mutual information: NMI = 2*I(X;Y) / (H(X) + H(Y))
    # Using max(0, ...) to ensure non-negative MI
    ent_1 = entropy(mat_1_canonical, nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2_canonical, nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)
    ent_joint = entropy(vcat(mat_1_canonical, mat_2_canonical), nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)

    mi = max(0, ent_1 + ent_2 - ent_joint)
    avg_entropy = (ent_1 + ent_2) / 2

    return mi / avg_entropy
end

function normalized_mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    return normalized_mutual_information(mat_1, mat_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    interaction_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3,base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the interaction information (II) between three datasets as II(X; Y; Z) = H(X) + H(Y) + H(Z) - H(X, Y) - H(X, Z) - H(Y, Z) + H(X, Y, Z)
  where:
  - H(X), H(Y), H(Z): Entropies of the individual datasets.
  - H(X, Y), H(X, Z), H(Y, Z): Pairwise joint entropies.
  - H(X, Y, Z): Joint entropy of all three datasets.


# Arguments
- `X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the first dataset.
- `Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the second dataset.
- `Z::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the third dataset.
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

# Returns
- `Real`: The computed interaction information II(X; Y; Z) between the datasets X, Y and Z.

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D
z = rand(1, 100)  # 100 points in 1D

# Using k-NN method
ii = interaction_information(x, y, z, method="knn", k=5, verbose=true)

# Using histogram method
ii = interaction_information(x, y, z, method="histogram", nbins=10)

# Using invariant method
ii = interaction_information(x, y, z, method="inv", k=3)
"""
function interaction_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    # Preprocessing: normalize data layout and extract shapes
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    mat_3_canonical = ensure_columns_are_points(mat_3, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)
    shape3 = get_shape(mat_3_canonical)

    # Validation
    validate_same_num_points([shape1, shape2, shape3])
    validate_dimensions_equal_one([shape1, shape2, shape3])
    verbose && log_computation_info([shape1, shape2, shape3], base)

    # Compute interaction information: II(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)
    ent_1 = entropy(mat_1_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_3 = entropy(mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_12 = entropy(vcat(mat_1_canonical, mat_2_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_13 = entropy(vcat(mat_1_canonical, mat_3_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_23 = entropy(vcat(mat_2_canonical, mat_3_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_123 = entropy(vcat(mat_1_canonical, mat_2_canonical, mat_3_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)

    return ent_1 + ent_2 + ent_3 - ent_12 - ent_13 - ent_23 + ent_123
end

function interaction_information(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    mat_3 = vector_to_matrix(array_3)
    return interaction_information(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end
                                                                

"""
    information_quality_ratio(X::Matrix{<:Real}, Y::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the Information Quality Ratio (IQR) between two datasets using the formula:
  IQR(X; Y) = (H(X) + H(Y) - H(X, Y)/H(X)
  where:
  - H(X): Entropy of the first dataset (X).
  - H(Y): Entropy of the second dataset (Y).
  - H(X, Y): Joint entropy of the two datasets.

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset.
- `Y::Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset.
- `method::String = "inv"` (optional): The method to use for entropy and mutual information computation. Options are:
  - `"knn"`: k-Nearest Neighbors based estimation.
  - `"histogram"`: Histogram-based estimation.
  - `"inv"`: Invariant estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computations. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The Information Quality Ratio (IQR), defined as:

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D

# Using k-NN method
iqr = information_quality_ratio(x, y, method="knn", k=5, verbose=true)

# Using histogram method
iqr = information_quality_ratio(x, y, method="histogram", nbins=10)

# Using invariant method
iqr = information_quality_ratio(x, y, method="inv", k=3)
"""

function information_quality_ratio(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    # Preprocessing: normalize data layout and extract shapes
    mat_1_canonical = ensure_columns_are_points(mat_1, dim)
    mat_2_canonical = ensure_columns_are_points(mat_2, dim)
    shape1 = get_shape(mat_1_canonical)
    shape2 = get_shape(mat_2_canonical)

    # Validation
    validate_same_num_points([shape1, shape2])
    validate_dimensions_equal_one([shape1, shape2])
    verbose && log_computation_info([shape1, shape2], base)

    # Compute information quality ratio: IQR(X;Y) = I(X;Y) / H(X)
    ent_1 = entropy(mat_1_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_joint = entropy(vcat(mat_1_canonical, mat_2_canonical), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)

    mi = ent_1 + ent_2 - ent_joint
    return mi / ent_1
end

function information_quality_ratio(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    return information_quality_ratio(mat_1, mat_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end



