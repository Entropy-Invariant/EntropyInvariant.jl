# Partial Information Decomposition (PID) Functions
#
# This file contains implementations for decomposing information relationships
# into non-overlapping components based on the Partial Information Decomposition
# framework.
#
# Functions included:
# - redundancy: Shared information between variables: min(I(X;Z), I(Y;Z))
# - unique: Unique information each variable provides
# - synergy: Combined information beyond individual contributions
#
# These functions are used to decompose multivariate information into:
# - Redundant information (shared by both)
# - Unique information (specific to each variable)
# - Synergistic information (only available jointly)

                                                                

"""
    redundancy(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the redundancy of information shared by two datasets X and Y about a third dataset Z using the mutual information between I(X; Z) and I(Y; Z): R(X, Y; Z) = min(I(X; Z), I(Y; Z))

# Arguments
- `X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset  X.
- `Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset Y.
- `Z::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the third dataset Z.
- `method::String = "inv"` (optional): The method to use for mutual information computation. Options are:
  - `"knn"`: k-Nearest Neighbors based mutual information estimation.
  - `"histogram"`: Histogram-based mutual information estimation.
  - `"inv"`: Invariant mutual information estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for mutual information computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The computed redundancy, defined as the minimum of the mutual information between  X and Z, and Y and Z:

# Behavior
- Depending on the specified `method`, the appropriate mutual information estimation function (`mutual_information`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D
z = rand(1, 100)  # 100 points in 1D

# Using k-NN method
redund = redundancy(x, y, z, method="knn", k=5, verbose=true)

# Using histogram method
redund = redundancy(x, y, z, method="histogram", nbins=10)

# Using invariant method
redund = redundancy(x, y, z, method="inv", k=3)
"""
function redundancy(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
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

    # Compute redundancy: R(X,Y;Z) = min(I(X;Z), I(Y;Z))
    mi_xz = mutual_information(mat_1_canonical, mat_3_canonical, k=k, method=method, nbins=nbins, base=base, degenerate=degenerate, dim=2)
    mi_yz = mutual_information(mat_2_canonical, mat_3_canonical, k=k, method=method, nbins=nbins, base=base, degenerate=degenerate, dim=2)

    return min(mi_xz, mi_yz)
end

function redundancy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    mat_3 = vector_to_matrix(array_3)
    return redundancy(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end


"""
    unique(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Tuple{Real, Real}

Compute the unique information that two datasets X and Y contribute individually about a third dataset Z as:
U(X; Z) = I(X; Z) - R(X, Y; Z)
U(Y; Z) = I(Y; Z) - R(X, Y; Z)
where:
  - R(X, Y; Z) = min(I(X; Z), I(Y; Z)) si the Redundancy of  X and Y about Z.

# Arguments
- `X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset X.
- `Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset Y.
- `Z::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the third dataset Z.
- `method::String = "inv"` (optional): The method to use for mutual information computation. Options are:
  - `"knn"`: k-Nearest Neighbors based mutual information estimation.
  - `"histogram"`: Histogram-based mutual information estimation.
  - `"inv"`: Invariant mutual information estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for mutual information computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Tuple{Real, Real}`: A tuple (U(X; Z), U(Y; Z)).
# Behavior
- Depending on the specified `method`, the appropriate mutual information estimation function (`mutual_information`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D
z = rand(1, 100)  # 100 points in 1D

# Using k-NN method
unique_x, unique_y = unique(x, y, z, method="knn", k=5, verbose=true)

# Using histogram method
unique_x, unique_y = unique(x, y, z, method="histogram", nbins=10)

# Using invariant method
unique_x, unique_y = unique(x, y, z, method="inv", k=3)

"""
function unique(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Tuple{Real, Real}
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

    # NOTE: +1 correction factor added to mutual information values
    # This ensures positive unique information when redundancy equals MI
    # Prevents numerical instability in edge cases
    mi_xz = mutual_information(mat_1_canonical, mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) + 1
    mi_yz = mutual_information(mat_2_canonical, mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) + 1

    # Compute unique information: U(X;Z) = I(X;Z) - R(X,Y;Z)
    redundancy_xy_z = min(mi_xz, mi_yz)
    unique_x = mi_xz - redundancy_xy_z
    unique_y = mi_yz - redundancy_xy_z

    return unique_x, unique_y
end

function unique(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Tuple{Real, Real}
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    mat_3 = vector_to_matrix(array_3)
    return unique(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end


"""
    synergy(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the synergy between two dataset sX and Y regarding their shared information about a third dataset Z using the formula:
  S(X, Y; Z) = I(X, Y; Z) - U(X; Z) - U(Y; Z) - R(X, Y; Z)
  where:
  - I(X, Y; Z): Conditional mutual information of X and Y about Z.
  - U(X; Z): Unique information that X contributes about Z.
  - U(Y; Z): Unique information that Y contributes about Z.
  - R(X, Y; Z): Redundancy of X and Y about Z. 

# Arguments
- `X:Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset ( X ).
- `Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset ( Y ).
- `Z::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the third dataset ( Z ).
- `method::String = "inv"` (optional): The method to use for mutual and conditional mutual information computation. Options are:
  - `"knn"`: k-Nearest Neighbors based estimation.
  - `"histogram"`: Histogram-based estimation.
  - `"inv"`: Invariant estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for information computations. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the datasets and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The computed synergy  S(X, Y; Z), representing the additional information that X and Y provide about Z together, beyond their redundancy and unique contributions.

# Behavior
 - The combined dimensions (rows) of all three datasets must equal 3.
- Calls `conditional_mutual_information`, `unique`, and `redundancy` functions to compute the respective components.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D
z = rand(1, 100)  # 100 points in 1D

# Using k-NN method
syn = synergy(x, y, z, method="knn", k=5, verbose=true)

# Using histogram method
syn = synergy(x, y, z, method="histogram", nbins=10)

# Using invariant method
syn = synergy(x, y, z, method="inv", k=3)
"""

function synergy(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
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

    # NOTE: +1 correction factors added to CMI and redundancy
    # This compensates for the +1 offset in unique() function
    # Maintains consistency across PID (Partial Information Decomposition) calculations
    cmi_xyz = conditional_mutual_information(mat_1_canonical, mat_2_canonical, mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) + 1
    unique_x, unique_y = unique(mat_1_canonical, mat_2_canonical, mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    redundancy_xy_z = redundancy(mat_1_canonical, mat_2_canonical, mat_3_canonical, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) + 1

    # Compute synergy: S(X,Y;Z) = I(X,Y;Z) - U(X;Z) - U(Y;Z) - R(X,Y;Z)
    return cmi_xyz - unique_x - unique_y - redundancy_xy_z
end

function synergy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    # Convert vectors to matrices
    mat_1 = vector_to_matrix(array_1)
    mat_2 = vector_to_matrix(array_2)
    mat_3 = vector_to_matrix(array_3)
    return synergy(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
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

