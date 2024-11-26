module EntropyInfo

# Import specific functions from StatsBase, NearestNeighbors and, SpecialFunctions
import StatsBase: median, mean
import NearestNeighbors: KDTree, knn
import SpecialFunctions: gamma, digamma

export entropy, conditional_entropy, mutual_information, conditional_mutual_information, normalized_mutual_information, interaction_information, redundancy, unique, synergy, information_quality_ratio

const e = 2.718281828459045  #

"""
    nn1(array::Vector{<:Real}) -> Vector{<:Real}

Calculate distances between neighboring elements in a vector.

# Arguments
- `array::Vector{<:Real}`: A vector of real numbers.

# Returns
- `Vector{<:Real}`: A vector containing the minimum distances between each element and its neighboring elements.

# Throws
- `ArgumentError`: If the input array contains fewer than two elements.

# Example
```julia
nn1([1.0, 3.0, 7.0])
```
# returns [2.0, 2.0, 4.0]
"""
function nn1(array::Vector{<:Real})
    m::Int = length(array)
    if m < 2
        throw(ArgumentError("Input array must contain more than one element"))
    end
    all_dist::Vector{<:Real} = zeros(Real, m)
    # Calculate distances for interior elements
    for i in 2:(m-1)
        all_dist[i] = min(abs(array[i] - array[i-1]), abs(array[i] - array[i+1]))
    end
    # Calculate distances for boundary elements
    all_dist[1] = abs(array[1] - array[2])
    all_dist[end] = abs(array[m-1] - array[m])
    return all_dist
end

"""
    hist1d(data::AbstractVector, nbins::Int) -> Vector{Int}

Compute a one-dimensional histogram from the given data.

# Arguments
- `data::AbstractVector`: A vector of numeric values to be binned.
- `nbins::Int`: The number of bins to divide the range of `data` into.

# Returns
- `Vector{Int}`: A vector where each element represents the count of data points within the corresponding bin.

# Behavior
- The function calculates the minimum and maximum values of the input `data`.
- The range between these values is divided into `nbins` evenly spaced bins, with edges computed using `range`.
- Each data point is assigned to a bin if it falls within the range of that bin. The bin edges are inclusive on the lower end and exclusive on the upper end, except for the last bin, which is inclusive on both ends to ensure all data points are accounted for."""
function hist1d(data, nbins)
    min_val, max_val = minimum(data), maximum(data)
    bin_edges = range(min_val, stop=max_val, length=nbins+1)
    counts = zeros(Int, nbins)
    for value in data
        for i in 1:nbins
            if value >= bin_edges[i] && value < bin_edges[i+1]
                counts[i] += 1
                break
            end
        end
    end
    return counts
end

"""
    hist2d(x::AbstractVector, y::AbstractVector, nbins::Int) -> Matrix{Int}

Compute a two-dimensional histogram from the given `x` and `y` data.

# Arguments
- `x::AbstractVector`: A vector of numeric values representing the first dimension.
- `y::AbstractVector`: A vector of numeric values representing the second dimension. Must be the same length as `x`.
- `nbins::Int`: The number of bins for both dimensions (same number of bins is used for x and y).

# Returns
- `Matrix{Int}`: A 2D matrix where each element represents the count of data points within the corresponding 2D bin.

# Behavior
- The function calculates the minimum and maximum values for `x` and `y`.
- It defines `nbins` evenly spaced bins for each dimension using `range`.
- For each data point `(x[k], y[k])`, the function determines the appropriate bin indices `(i, j)` and increments the corresponding entry in the histogram matrix.
- Points outside the range of the bins are ignored.

"""
function hist2d(x, y, nbins)
    # Find the minimum and maximum for each dimension
    min_x, max_x = minimum(x), maximum(x)
    min_y, max_y = minimum(y), maximum(y)
    
    # Define bin edges for each dimension
    bin_edges_x = range(min_x, stop=max_x, length=nbins+1)
    bin_edges_y = range(min_y, stop=max_y, length=nbins+1)

    # Initialize counts
    counts = zeros(Int, nbins, nbins)

    # Count occurrences in each bin
    for k in 1:length(x)
        # Find the appropriate bin index for x[k] and y[k]
        i = findfirst(t -> t >= x[k], bin_edges_x) - 1
        j = findfirst(t -> t >= y[k], bin_edges_y) - 1
        
        # Ensure the index is within the bounds of the counts array
        if 1 <= i <= nbins && 1 <= j <= nbins
            counts[i, j] += 1
        end
    end
    return counts
end


"""
    hist3d(x::AbstractVector, y::AbstractVector, z::AbstractVector, nbins::Int) -> Array{Int, 3}

Compute a three-dimensional histogram from the given `x`, `y`, and `z` data.

# Arguments
- `x::AbstractVector`: A vector of numeric values representing the first dimension.
- `y::AbstractVector`: A vector of numeric values representing the second dimension. Must be the same length as `x`.
- `z::AbstractVector`: A vector of numeric values representing the third dimension. Must be the same length as `x` and `y`.
- `nbins::Int`: The number of bins for all three dimensions (same number of bins is used for `x`, `y`, and `z`).

# Returns
- `Array{Int, 3}`: A 3D array where each element represents the count of data points within the corresponding 3D bin.

# Behavior
- The function calculates the minimum and maximum values for `x`, `y`, and `z`.
- It defines `nbins` evenly spaced bins for each dimension using `range`.
- For each data point `(x[k], y[k], z[k])`, the function determines the appropriate bin indices `(i, j, l)` and increments the corresponding entry in the histogram array.
- Points outside the range of the bins are ignored.
"""
function hist3d(x, y, z, nbins)
    # Find min and max for each dimension
    min_x, max_x = minimum(x), maximum(x)
    min_y, max_y = minimum(y), maximum(y)
    min_z, max_z = minimum(z), maximum(z)
    
    # Define bin edges for each dimension
    bin_edges_x = range(min_x, stop=max_x, length=nbins+1)
    bin_edges_y = range(min_y, stop=max_y, length=nbins+1)
    bin_edges_z = range(min_z, stop=max_z, length=nbins+1)
    
    # Initialize counts array
    counts = zeros(Int, nbins, nbins, nbins)
    
    # Count occurrences in each bin
    for k in 1:length(x)
        # Find the appropriate bin index for x[k], y[k], and z[k]
        i = findfirst(t -> t >= x[k], bin_edges_x) - 1
        j = findfirst(t -> t >= y[k], bin_edges_y) - 1
        l = findfirst(t -> t >= z[k], bin_edges_z) - 1
        
        # Ensure the index is within the bounds of the counts array
        if 1 <= i <= nbins && 1 <= j <= nbins && 1 <= l <= nbins
            counts[i, j, l] += 1
        end
    end
    return counts
end


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
    if dim == 1
        mat_ = Matrix{Float64}(transpose(mat_))
    end
    n = length(mat_[1,:]) #nb de points
    d = length(mat_[:,1]) #dimension
    
    if verbose
        println("Number of points: $n")
        println("Dimensions: $d")
        println("Base: $base")
    end
    if d > 3
        throw(ArgumentError("Maximum dimension for histogram method is 3."))
    end
    # Compute histogram
    if d == 1
        weights = hist1d(mat_, nbins)
    elseif d == 2
        weights = hist2d(mat_[1,:], mat_[2,:], nbins)
    else
        weights = hist3d(mat_[1,:], mat_[2,:], mat_[3,:], nbins)
    end
    # Convert counts to probability distribution
    probs = weights ./ sum(weights)
    ent = 0.0
    # Compute entropy
    for i in probs
        if i != 0
            ent += i.*log(i)
        end
    end
    return -ent
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
    if dim == 1
        mat_ = Matrix{Float64}(transpose(mat_))
    end
    n = length(mat_[1,:]) #nb de points
    d = length(mat_[:,1]) #dimension
    
    if verbose
        println("Number of points: $n")
        println("Dimensions: $d")
        println("Base: $base")
    end
    noise = 0
    if degenerate
        noise = 1
    end
    kdtree = KDTree(mat_)
    idxs, dists = knn(kdtree, mat_, k+1, true)
    dists_k = [i[k+1] for i in dists]
    log_dists_k = []
    for i in dists_k
        if i != 0 #log(0) not define 
            push!(log_dists_k, log(i+noise))
        end
    end
    volume_unit_ball = (pi^(0.5*d)) / gamma(.5*d + 1)
    ent = d*mean(log_dists_k)+log(volume_unit_ball)+digamma(length(log_dists_k))-digamma(k)
    return ent*log(base, e)
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
    if dim == 1
        mat_ = Matrix{Float64}(transpose(mat_))
    end
    n = length(mat_[1,:]) #nb de points
    d = length(mat_[:,1]) #dimension
    
    if verbose
        println("Number of points: $n")
        println("Dimensions: $d")
        println("Base: $base")
    end
    noise = 0
    if degenerate
        noise = 1
    end
    r_ = zeros(Real, d)
    for i in 1:d
        skip_zero = filter(x -> x!= 0 , mat_[i,:]) # if null value more than half the size the vector the median is 0
        r_[i] = median(sort(nn1(sort(skip_zero))))*length(skip_zero)
        mat_[i,:] /= r_[i] # then we can't divide by zero
    end
    kdtree = KDTree(mat_)
    idxs, dists = knn(kdtree, mat_, k+1, true)
    dists_k = [i[k+1] for i in dists]
    log_dists_k = []
    for i in dists_k
        if i != 0 #log(0) not define 
            push!(log_dists_k, log(i+noise))
        end
    end
    volume_unit_ball = (pi^(0.5*d)) / gamma(.5*d + 1)
    ent = d*mean(log_dists_k)+log(volume_unit_ball)+digamma(length(log_dists_k))-digamma(k)
    return ent*log(base, e)
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
        return entropy_hist(mat_, nbins=nbins, dim=dim, verbose=verbose)
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


"""
    conditional_entropy(X::Matrix{<:Real}, Y::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the conditional entropy of a dataset X given a conditioning dataset Y as  H(X | Y) = H(X, Y) - H(Y) where:
  - H(X, Y): Joint entropy of X and Y.
  - H(Y): Entropy of the conditioning dataset Y.


# Arguments
- `X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the primary dataset.
- `Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point and each row represents a dimension for the conditioning dataset.
- `method::String = "inv"` (optional): The method to use for entropy computation. Options are:
  - `"knn"`: k-Nearest Neighbors based entropy estimation.
  - `"histogram"`: Histogram-based entropy estimation.
  - `"inv"`: Invariant entropy estimation (default).
- `nbins::Int = 10` (optional): The number of bins for the histogram method. Ignored for other methods. Defaults to 10.
- `k::Int = 3` (optional): The number of neighbors for the k-NN or invariant method. Ignored for the histogram method. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for entropy computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the dataset and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for the k-NN or invariant method to handle degenerate cases. Ignored for the histogram method. Defaults to `false`.
- `dim::Int = 1` (optional): Indicates whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1 (data in rows).

# Returns
- `Real`: The computed conditional entropy H(X | Y).

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D

# Using k-NN method
conditional_ent = conditional_entropy(x, y, method="knn", k=5, verbose=true)

# Using histogram method
conditional_ent = conditional_entropy(x, y, method="histogram", nbins=10)

# Using invariant method
conditional_ent = conditional_entropy(x, y, method="inv", k=3)
"""
function conditional_entropy(mat_::Matrix{<:Real}, cond_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    if dim == 1
        mat_ = Matrix{Float64}(transpose(mat_))
        cond_ = Matrix{Float64}(transpose(cond_))
    end
    n1 = length(mat_[1,:]) #nb de points
    n2 = length(cond_[1,:]) #nb de points
    d1 = length(mat_[:,1]) #dimension
    d2 = length(cond_[:,1]) #dimension
    if n1 != n2
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != 1
        throw(ArgumentError("The total dimension should be 2"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2)")
        println("Base: $base")
    end
    ent_ = entropy(mat_, k=k, method=method, nbins=nbins,  base=base, degenerate=degenerate, dim=2)
    ent_joint_ = entropy(vcat(mat_,cond_), method=method, nbins=nbins,  k=k, base=base, degenerate=degenerate, dim=2)
    return ent_joint_-ent_
end

function conditional_entropy(array_::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_ = reshape(array_, length(array_), 1)
    cond_ = reshape(cond_, length(cond_), 1)
    return conditional_entropy(array_, cond_, method=method, k=k, nbins=nbins, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    mutual_information(X::Matrix{<:Real}, Y::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the mutual information between two datasets as I(X; Y) = H(X) + H(Y) - H(X, Y)
  where:
  - H(X): Entropy of the first dataset X.
  - H(Y): Entropy of the second dataset Y.
  - H(X, Y): Joint entropy of X and Y.

# Arguments
- X::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the first dataset.
- Y::Vector or Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension for the second dataset.
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
- `Real`: The computed mutual information I(X; Y) between the datasets X and Y.

# Behavior
- Depending on the specified `method`, the appropriate entropy estimation function (`entropy_knn`, `entropy_hist`, `entropy_inv`) is called.

# Example
x = rand(1, 100)  # 100 points in 1D
y = rand(1, 100)  # 100 points in 1D

# Using histogram kNN method
mi = mutual_information(x, y, method="knn", k=5, verbose=true)

# Using histogram method
mi = mutual_information(x, y, method="histogram", nbins=10)

# Using invariant method
mi = mutual_information(x, y, method="inv", k=3)
"""
function mutual_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
    end
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    if n1 != n2
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != 1
        throw(ArgumentError("The total dimension should be 2"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2)")
        println("Base: $base")
    end
    ent_1 = entropy(mat_1, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_12 = entropy(vcat(mat_1,mat_2), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    return ent_1+ent_2-ent_12
end

function mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    mat_1 = reshape(array_1, length(array_1), 1)
    mat_2 = reshape(array_2, length(array_2), 1)
    return mutual_information(mat_1, mat_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end
                                


"""
    conditional_mutual_information(X::Matrix{<:Real}, Y::Matrix{<:Real}, Z::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

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
function conditional_mutual_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, cond_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
        cond_ = Matrix{Float64}(transpose(cond_))
    end 
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    n3 = length(cond_[1,:]) #nb de points                      
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    d3 = length(cond_[:,1]) #dimension
    if (n1 != n2) | (n1 != n2) | (n2 != n3) 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != d3 != 1
        throw(ArgumentError("The total dimension should be 3"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2+d3)")
        println("Base: $base")
    end
    ent_cond_ = entropy(cond_, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_cond1_ = entropy(vcat(mat_1, cond_), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_cond2_ = entropy(vcat(mat_2, cond_), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_cond12_ = entropy(vcat(mat_1, mat_2, cond_), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    return ent_cond1_+ent_cond2_-ent_cond12_-ent_cond_
end

function conditional_mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    cond_ = reshape(cond_, length(cond_), 1)
    return conditional_mutual_information(array_1, array_2, cond_, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    normalized_mutual_information(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}; method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Compute the normalized mutual information (NMI) between two datasets as  NMI(X, Y) = max(0, H(X) + H(Y) - H(X, Y))/(H(X) + H(Y)
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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
    end    
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points                     
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    if n1 != n2 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != 1
        throw(ArgumentError("The total dimension should be 2"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2)")
        println("Base: $base")
    end
    ent_1 = entropy(mat_1, nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2, nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)
    ent_12 = entropy(vcat(mat_1, mat_2), nbins=nbins, method=method, k=k, base=base, degenerate=degenerate, dim=2)
    return max(0, ent_1+ent_2-ent_12)/(ent_1+ent_2)
end

function normalized_mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    return normalized_mutual_information(array_1, array_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
        mat_3 = Matrix{Float64}(transpose(mat_3))
    end    
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    n3 = length(mat_3[1,:]) #nb de points                      
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    d3 = length(mat_3[:,1]) #dimension
    if (n1 != n2) | (n1 != n2) | (n2 != n3) 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != d3 != 1
        throw(ArgumentError("The total dimension should be 3"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2+d3)")
        println("Base: $base")
    end
    ent_1_ = entropy(mat_1, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2_ = entropy(mat_2, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_3_ = entropy(mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_12_ = entropy(vcat(mat_1, mat_2), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_13_ = entropy(vcat(mat_1, mat_3), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_23_ = entropy(vcat(mat_2, mat_3), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_123_ = entropy(vcat(mat_1, mat_2, mat_3), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    return ent_1_+ent_2_+ent_3_-ent_12_-ent_13_-ent_23_+ent_123_
end

function interaction_information(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return interaction_information(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end
                                                                

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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
        mat_3 = Matrix{Float64}(transpose(mat_3))
    end    
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    n3 = length(mat_3[1,:]) #nb de points                      
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    d3 = length(mat_3[:,1]) #dimension
    if (n1 != n2) | (n1 != n2) | (n2 != n3) 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != d3 != 1
        throw(ArgumentError("The total dimension should be 3"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2+d3)")
        println("Base: $base")
    end
    mi_xz = mutual_information(mat_1, mat_3, k=k, method=method, nbins=nbins, base=base, degenerate=degenerate, dim=2)
    mi_yz = mutual_information(mat_2, mat_3, k=k, method=method, nbins=nbins, base=base, degenerate=degenerate, dim=2)
    return min(mi_xz, mi_yz)
end

function redundancy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    array_3 = reshape(array_3, length(array_3), 1)
    return redundancy(array_1, array_2, array_3, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
        mat_3 = Matrix{Float64}(transpose(mat_3))
    end    
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    n3 = length(mat_3[1,:]) #nb de points                     
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    d3 = length(mat_3[:,1]) #dimension
    if (n1 != n2) | (n1 != n2) | (n2 != n3) 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != d3 != 1
        throw(ArgumentError("The total dimension should be 3"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2+d3)")
        println("Base: $base")
    end
    mi_xz = mutual_information(mat_1, mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)+1
    mi_yz = mutual_information(mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)+1
    re_xy_z = min(mi_xz, mi_yz)
    uni_x = mi_xz-re_xy_z
    uni_y = mi_yz-re_xy_z
    return uni_x, uni_y
end

function unique(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Tuple{Real, Real}
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return unique(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
        mat_3 = Matrix{Float64}(transpose(mat_3))
    end    
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points
    n3 = length(mat_3[1,:]) #nb de points                     
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    d3 = length(mat_3[:,1]) #dimension
    if (n1 != n2) | (n1 != n2) | (n2 != n3) 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != d3 != 1
        throw(ArgumentError("The total dimension should be 3"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2+d3)")
        println("Base: $base")
    end
    cmi_xyz = conditional_mutual_information(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)+1
    uni_x, uni_y = unique(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    re_xy_z = redundancy(mat_1, mat_2, mat_3, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)+1
    return cmi_xyz-uni_x-uni_y-re_xy_z
end

function synergy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return synergy(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
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
    if dim == 1
        mat_1 = Matrix{Float64}(transpose(mat_1))
        mat_2 = Matrix{Float64}(transpose(mat_2))
    end 
    n1 = length(mat_1[1,:]) #nb de points
    n2 = length(mat_2[1,:]) #nb de points                     
    d1 = length(mat_1[:,1]) #dimension
    d2 = length(mat_2[:,1]) #dimension
    if n1 != n2 
        throw(ArgumentError("Input arrays must contain the same number of points"))
    end
    if d1 != d2 != 1
        throw(ArgumentError("The total dimension should be 2"))
    end
    if verbose
        println("Number of points: $n1")
        println("Dimensions: $(d1+d2)")
        println("Base: $base")
    end
    ent_1 = entropy(mat_1, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_2 = entropy(mat_2, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    ent_12 = entropy(vcat(mat_1, mat_2), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
    return (ent_1+ent_2-ent_12)/ent_1
end

function information_quality_ratio(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    return information_quality_ratio(array_1, array_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    MI(a::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Matrix{<:Real}

Compute the mutual information (MI) matrix for all pairs of dimensions in the given dataset using the k-nearest neighbors (k-NN) invariant measure as: I(X_i; X_j) = H(X_i) + H(X_j) - H(X_i, X_j), where H(X_i) is the entropy of dimension i, and H(X_i, X_j) is the joint entropy of dimensions i and j.


# Arguments
- `a::Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension of the dataset.
- `k::Int = 3` (optional): The number of nearest neighbors to consider for k-NN estimation. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for MI computation. Defaults to the natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints additional information about the dataset and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances to handle degenerate cases. Defaults to `false`.
- `dim::Int = 1` (optional): Specifies whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1.

# Returns
- `Matrix{<:Real}`: A symmetric matrix M, where  M[i, j] represents the mutual information between the i-th and j-th dimensions of the dataset.

# Behavior
- The function computes mutual information for all pairs of dimensions in the dataset using a k-NN-based invariant measure.
- Steps:
  1. Normalize each dimension by dividing by its invariant measure (computed using the median distance to nearest neighbors).
  2. Compute marginal entropy for each dimension using k-NN.
  3. Compute joint entropy for each pair of dimensions.
  4. Compute mutual information for each pair

# Example
# Compute MI for a 3-dimensional dataset
data = rand(3, 100)  # 100 points in 3 dimensions
mi_matrix = MI(data, k=5, verbose=true)

# Compute MI for transposed data
data_t = rand(100, 3)  # Transposed dataset
mi_matrix = MI(data_t, k=3, dim=2)
"""
function MI(a::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Matrix{<:Real}
    if dim == 2
       a = Matrix{Float64}(transpose(a))
    end
    n = length(a[:,1])
    m = length(a[1,:])
    
    if verbose
        println("Number of points: $n")
        println("Dimensions: $m")
        println("Base: $base")
    end
    noise = 0
    if degenerate
        noise = 1
    end
    
    volume_unit_ball = [2.0 3.141592653589793] #dim = [1,2]
    log_volume_unit_ball = [log(i) for i in volume_unit_ball]

    d1 = 1
    d2 = 2
    k_1 = k+1

    log_volume_unit_ball_1 = log_volume_unit_ball[d1]
    log_volume_unit_ball_2 = log_volume_unit_ball[d2]

    dig_k = digamma(k)
    dig_n = digamma(n)

    # Compute the invariant measure for all dimensions
    all_ri = zeros(m)
    for i in 1:m
        all_ri[i] = median(sort(nn1(sort(a[:,i]))))*n
    end

    # Reshape and divide by invariant measure all dimensions
    # m-element Vector{Matrix{<:Real}} with 1×n Matrix{<:Real}
    all_a_ri = [reshape(a[:,i]/all_ri[i], 1, n) for i in 1:m]

    # Compute all marginal entropy KNN INV
    all_ent_i = zeros(m)
    for i in 1:m
        kdtree_i = KDTree(all_a_ri[i])

        idxs_i, dists_i = knn(kdtree_i, all_a_ri[i], k_1, true)

        dists_k_i = [i[k_1] for i in dists_i]

        log_dists_k_i = []
        for j in dists_k_i
            if j != 0 #log(0) non defini 
                push!(log_dists_k_i, log(j+noise))
            end
        end
        all_ent_i[i] = d1*mean(log_dists_k_i)+log_volume_unit_ball_1+dig_n-dig_k
    end
        
    # m-element m-element Vector{Matrix{<:Real}} with 2×n Matrix{<:Real}
    all_ij = [[vcat(all_a_ri[i], all_a_ri[j]) for i in 1:m] for j in 1:m]

    # Compute all double joint entropy KNN INV
    all_ent_ij = zeros(m,m)
    for i in 1:m
        #println(i)
        for j in 1:m
            if i <= j
                kdtree_ij = KDTree(all_ij[i][j])
                idxs_ij, dists_ij = knn(kdtree_ij, all_ij[i][j], k_1, true)

                dists_k_ij = [i[k_1] for i in dists_ij]
                log_dists_k_ij = []
                for k in dists_k_ij
                    if k != 0 #log(0) non defini
                        push!(log_dists_k_ij, log(k+noise))
                    end
                end
                all_ent_ij[i,j] = d2*mean(log_dists_k_ij)+log_volume_unit_ball_2+dig_n-dig_k
                all_ent_ij[j,i] = all_ent_ij[i,j]
            end
        end
    end
            
    # Compute MI
    all_mi_ij = zeros(m,m)
    for i in 1:m
        for j in 1:m
            if i <= j
                all_mi_ij[i,j] = all_ent_i[i]+all_ent_i[j]-all_ent_ij[i,j]
                all_mi_ij[j,i] = all_mi_ij[i,j]
            end
        end
    end
    return all_mi_ij*log(base, e)
end


"""
    CMI(X::Matrix{<:Real}, Z::Vector{<:Real}; base::Real = e, k::Int = 3, verbose::Bool = false, 
        degenerate::Bool = false, dim::Int = 1) -> Matrix{<:Real}

Compute the conditional mutual information (CMI) matrix for all pairs of dimensions in a dataset a, given a conditioning variable b, using the k-nearest neighbors (k-NN) invariant measure as CMI(X_i; X_j | Z) = H(X_i, Z) + H(X_j, Z) - H(X_i, X_j, Z) - H(Z)


# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a data point, and each row represents a dimension of the dataset.
- `Z::Vector{<:Real}`: A vector representing the conditioning variable.
- `base::Real = e` (optional): The logarithmic base for CMI computation. Defaults to the natural logarithm (`e`).
- `k::Int = 3` (optional): The number of nearest neighbors to consider for k-NN estimation. Defaults to 3.
- `verbose::Bool = false` (optional): If `true`, prints additional information about the dataset and computation process. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances to handle degenerate cases. Defaults to `false`.
- `dim::Int = 1` (optional): Specifies whether the data is organized in rows (`dim = 1`) or columns (`dim = 2`). Defaults to 1.

# Returns
- `Matrix{<:Real}`: A symmetric matrix C, where C[i, j] represents the conditional mutual information between the i-th and j-th dimensions of X, conditioned on Z.

# Behavior
- The function computes CMI for all pairs of dimensions in X, conditioned on Z, using a k-NN-based invariant measure.
- Steps:
  1. Normalize each dimension of X and Z using their respective invariant measures.
  2. Compute marginal entropy for each dimension of X.
  3. Compute joint entropy for each dimension of X and Z.
  4. Compute triple joint entropy for all pairs of dimensions in X, conditioned on Z.
  5. Calculate conditional mutual information.

# Example
# Compute CMI for a 3-dimensional dataset
data = rand(3, 100)  # 100 points in 3 dimensions
conditioning_var = rand(100)  # Conditioning variable
cmi_matrix = CMI(data, conditioning_var, k=5, verbose=true)

# Compute CMI for transposed data
data_t = rand(100, 3)  # Transposed dataset
cmi_matrix = CMI(data_t, conditioning_var, k=3, dim=2)
"""
function CMI(a::Matrix{<:Real}, b::Vector{<:Real}; base::Real = e, k::Int = 3, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Matrix{<:Real}
    if dim == 2
        a = Matrix{Float64}(transpose(a))
    end   
    n = length(a[:,1]) #number of points
    m = length(a[1,:]) #dimension
    
    if verbose
        println("Number of points: $n")
        println("Dimensions: $m")
        println("Base: $base")
    end
    noise = 0
    if degenerateoentro
        noise = 1
    end

    volume_unit_ball = [2.0 3.141592653589793 4.188790204786391] #dim = [1,2,3]
    log_volume_unit_ball = [log(i) for i in volume_unit_ball]

    d1 = 1
    d2 = 2
    d3 = 3
    k_1 = k+1

    log_volume_unit_ball_1 = log_volume_unit_ball[d1]
    log_volume_unit_ball_2 = log_volume_unit_ball[d2]
    log_volume_unit_ball_3 = log_volume_unit_ball[d3]

    dig_k = digamma(k)
    dig_n = digamma(n)


    # Compute the invariant measure for all peaks
    all_ri = zeros(m)
    for i in 1:m
        all_ri[i] = median(sort(nn1(sort(a[:,i]))))*n
    end

    # Compute the invatiant measure for the conditional variable
    rz = median(sort(nn1(sort(b))))*n
    # Reshape and divide by invariant measure all peaks
    # m-element Vector{Matrix{<:Real}} with 1×n Matrix{<:Real}
    all_a_ri = [reshape(a[:,i]/all_ri[i], 1, n) for i in 1:m]

    # Reshape and divide by invariant measure the conditional variable
    b_rz = reshape(b/rz, 1, n)
    # Compute marginal entropy KNN INV for the conditional variable
    kdtree_z = KDTree(b_rz)
    idxs_z, dists_z = knn(kdtree_z, b_rz, k_1, true)
    dists_k_z = [i[k_1] for i in dists_z]
    log_dists_k_z = []
    for i in dists_k_z
        if i != 0 #log(0) non defini 
            push!(log_dists_k_z, log(i+noise))
        end
    end
    ent_z = d1*mean(log_dists_k_z)+log_volume_unit_ball_1+dig_n-dig_k
    # Compute all marginal entropy KNN INV
    all_ent_i = zeros(m)
    for i in 1:m
        kdtree_i = KDTree(all_a_ri[i])

        idxs_i, dists_i = knn(kdtree_i, all_a_ri[i], k_1, true)

        dists_k_i = [i[k_1] for i in dists_i]

        log_dists_k_i = []
        for j in dists_k_i
            if j != 0 #log(0) non defini 
                push!(log_dists_k_i, log(j+noise))
            end
        end

        all_ent_i[i] = d1*mean(log_dists_k_i)+log_volume_unit_ball_1+dig_n-dig_k
    end

    # m-element Vector{Matrix{<:Real}} with 2×n Matrix{<:Real}
    all_j_iz = [vcat(all_a_ri[i], b_rz) for i in 1:m]

    # Compute all double joint entropy KNN INV
    all_ent_iz = zeros(m)
    for i in 1:m
        kdtree_iz = KDTree(all_j_iz[i])

        idxs_iz, dists_iz = knn(kdtree_iz, all_j_iz[i], k_1, true)

        dists_k_iz = [i[k_1] for i in dists_iz]
        log_dists_k_iz = []
        for j in dists_k_iz
            if j != 0 #log(0) non defini 
                push!(log_dists_k_iz, log(j+noise))
            end
        end

        all_ent_iz[i] = d2*mean(log_dists_k_iz)+log_volume_unit_ball_2+dig_n-dig_k
    end

    # m-element m-element Vector{Matrix{<:Real}} with 3×n Matrix{<:Real}
    all_j_ijz = [[vcat(all_a_ri[i], all_a_ri[j], b_rz) for i in 1:m] for j in 1:m]

    # Compute all triple joint entropy KNN INV
    all_ent_ijz = zeros(m,m)
    for i in 1:m
        #println(i)
        for j in 1:m
            if i <= j
                kdtree_ijz = KDTree(all_j_ijz[i][j])
                idxs_ijz, dists_ijz = knn(kdtree_ijz, all_j_ijz[i][j], k_1, true)

                dists_k_ijz = [i[k_1] for i in dists_ijz]
                log_dists_k_ijz = []
                for k in dists_k_ijz
                    if k != 0 #log(0) non defini
                        push!(log_dists_k_ijz, log(k+noise))
                    end
                end
                all_ent_ijz[i,j] = d3*mean(log_dists_k_ijz)+log_volume_unit_ball_3+dig_n-dig_k
                all_ent_ijz[j,i] = all_ent_ijz[i,j]
            end
        end
    end

    # Compute CMI
    all_cmi_ijz = zeros(m,m)        
    for i in 1:m
        for j in 1:m
            if i <= j
                all_cmi_ijz[i,j] = all_ent_iz[i]+all_ent_iz[j]-all_ent_ijz[i,j]-ent_z
                all_cmi_ijz[j,i] = all_cmi_ijz[i,j]
            end
        end
    end
    return all_cmi_ijz*log(base, e)
end
                                                                                                                                    
function CMI(a::Matrix{<:Real}, b::Matrix{<:Real}; base::Real = e, k::Int = 3, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Matrix{<:Real}
    n2 = length(b[:,1])
    d2 = length(b[1,:])
    if (n2 != 1) & (dim == 2) | ((d2 != 1) & (dim == 1))
        throw(ArgumentError("Conditional arrays must contain the same number of points in one dimension"))
    end
    return CMI(a, vec(b), base=base, k=k, verbose=verbose, degenerate=degenerate, dim=dim)
end

end
