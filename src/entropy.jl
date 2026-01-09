# Core entropy estimation functions for EntropyInvariant package

"""
    entropy_hist(X::Matrix{<:Real}; nbins::Int = 10, dim::Int = 1, verbose::Bool = false) -> Real

Compute the entropy of a dataset using a histogram-based method.

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension.
- `nbins::Int = 10` (optional): The number of bins to use for the histogram. Defaults to 10.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).
- `verbose::Bool = false` (optional): If `true`, prints information about the dataset (number of points, dimensions, etc.). Defaults to `false`.

# Returns
- `Real`: The computed entropy of the dataset.

# Behavior
- The function computes the entropy of the dataset based on the histogram of the data:
  - If `dim = 1`, the matrix is transposed to organize data points as columns.
  - The function supports up to 3 dimensions (rows) for the dataset; an error is raised for higher dimensions.
  - Depending on the dimensionality:
    - For 1D data, the function uses `hist1d`.
    - For 2D data, it uses `hist2d`.
    - For 3D data, it uses `hist3d`.
- The histogram counts are normalized to obtain a probability distribution.
- The entropy is computed as: H = -Sum(p_i log(p_i) ), where p_i is the probability of each bin.
"""
function entropy_hist(mat_::Matrix{<:Real}; nbins::Int = 10, dim::Int = 1, base::Real = e, verbose::Bool = false)::Real
    # Preprocessing: normalize data layout and extract shape
    mat = ensure_columns_are_points(mat_, dim)
    shape = get_shape(mat)
    verbose && log_computation_info(shape, base)

    # Histogram method only supports dimensions 1-3
    if shape.num_dimensions > 3
        throw(ArgumentError("Maximum dimension for histogram method is 3."))
    end

    # Compute histogram based on dimensionality
    weights = if shape.num_dimensions == 1
        hist1d(mat, nbins)
    elseif shape.num_dimensions == 2
        hist2d(mat[1,:], mat[2,:], nbins)
    else  # dimension == 3
        hist3d(mat[1,:], mat[2,:], mat[3,:], nbins)
    end

    # Convert counts to probability distribution
    probs = weights ./ sum(weights)

    # Compute entropy: H = -Σ p(x) log(p(x))
    entropy_nats = 0.0
    for p in probs
        if p != 0
            entropy_nats -= p * log(p)
        end
    end

    # BUG FIX: Original code ignored the 'base' parameter
    # Now properly converts from nats to specified base
    return convert_to_base(entropy_nats, base)
end

"""
    entropy_knn(X::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Estimate the entropy of a dataset using the k-nearest neighbors (k-NN) method.

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension.
- `k::Int = 3` (optional): The number of nearest neighbors to consider. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints information about the dataset (number of points, dimensions, and base). Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds a small noise term to distances to handle degenerate cases. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The estimated entropy of the dataset.
"""
function entropy_knn(mat_::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    # Preprocessing: normalize data layout and extract shape
    mat = ensure_columns_are_points(mat_, dim)
    shape = get_shape(mat)
    verbose && log_computation_info(shape, base)

    # K-NN computation (without invariant measure normalization)
    noise = degenerate ? 1 : 0
    knn_result = compute_knn_distances(mat, k)
    log_dists = extract_nonzero_log_distances(knn_result.kth_distances, noise)

    # Entropy computation using Kraskov formula
    entropy_nats = compute_knn_entropy_nats(log_dists, shape.num_dimensions, k)
    return convert_to_base(entropy_nats, base)
end

"""
    entropy_inv(X::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Estimate the entropy of a dataset using the invariant method.

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension.
- `k::Int = 3` (optional): The number of nearest neighbors to consider. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints information about the dataset (number of points, dimensions, and base). Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds a small noise term to distances to handle degenerate cases. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The estimated entropy of the dataset.
"""
function entropy_inv(mat_::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    # Preprocessing: normalize data layout and extract shape
    mat = ensure_columns_are_points(mat_, dim)
    shape = get_shape(mat)
    verbose && log_computation_info(shape, base)

    # Invariant measure normalization
    normalized_mat = normalize_by_invariant_measure(mat)

    # K-NN computation
    noise = degenerate ? 1 : 0
    knn_result = compute_knn_distances(normalized_mat, k)
    log_dists = extract_nonzero_log_distances(knn_result.kth_distances, noise)

    # Entropy computation using Kraskov formula
    entropy_nats = compute_knn_entropy_nats(log_dists, shape.num_dimensions, k)
    return convert_to_base(entropy_nats, base)
end

"""
    entropy(X::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the entropy of a dataset using one of several methods.

# Arguments
- `X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension.
- `method::String = "inv"` (optional): The method to use for entropy computation. Options are:
  - `"knn"`: k-Nearest Neighbors (k-NN) based entropy estimation.
  - `"histogram"`: Histogram-based entropy estimation.
  - `"inv"`: Invariant entropy estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the dataset and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The computed entropy of the dataset.

# Behavior
- The function chooses the entropy estimation method based on the `method` argument:
  1. **k-NN Method** (`method = "knn"`):
     - Estimates entropy based on nearest-neighbor distances.
  2. **Histogram Method** (`method = "histogram"`):
     - Estimates entropy using a histogram of the data.
  3. **Invariant Method** (`method = "inv"`):
     - Estimates entropy with invariant scaling properties using nearest neighbors.
- The appropriate entropy function (`entropy_knn`, `entropy_hist`, or `entropy_inv`) is called based on the specified method.

# Example
# Using k-NN method
data = rand(1, 100)  # 100 points in 1 dimension
println("Entropy (k-NN): ", entropy(data, method="knn", k=5, verbose=true))

# Using histogram method
data = rand(100)  # 100 points in 1 dimension
println("Entropy (Histogram): ", entropy(data, method="histogram", nbins=10) )

# Using invariant method
println("Entropy (Invariant): ", entropy(data, method="inv", k=3))
"""
function entropy(mat_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    if method == "knn"
        return entropy_knn(mat_, k=k, verbose=verbose, degenerate=degenerate, base=base, dim=dim)
    elseif method == "histogram"
        return entropy_hist(mat_, nbins=nbins, dim=dim, verbose=verbose, base=base)
    elseif method == "inv"
        return entropy_inv(mat_, k=k, verbose=verbose, degenerate=degenerate, base=base, dim=dim)
    else
        throw(ArgumentError("Invalid method: $method. Choose either 'inv', 'knn' or 'histogram'."))
    end
end

function entropy(array::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    mat_ = reshape(array, length(array), 1)
    return entropy(mat_, k=k, verbose=verbose, degenerate=degenerate, base=base, dim=dim, method=method, nbins=nbins)
end
