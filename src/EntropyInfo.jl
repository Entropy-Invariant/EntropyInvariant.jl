module EntropyInfo

# Import specific functions from StatsBase, NearestNeighbors and, SpecialFunctions
import StatsBase: median, mean
import NearestNeighbors: KDTree, knn
import SpecialFunctions: gamma, digamma

export entropy, conditional_entropy, mutual_information, conditional_mutual_information, normalized_mutual_information, interaction_information, redundancy, unique, synergy, information_quality_ratio, MI2D, CMI2D

const e = exp(1)

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

function entropy_hist(mat_::Matrix{<:Real}; nbins::Int = 10, dim::Int = 1, verbose::Bool = false)::Real
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
    entropy(mat_::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1) -> Real

Estimate the entropy of a dataset using the k-nearest neighbors method.

# Arguments
- `mat_::Matrix{<:Real}`: Input matrix where each column represents a sample, and rows represent variables.
- `k::Int`: Number of neighbors for k-NN entropy estimation. Default is `3`.
- `nbins::Int`: Number of bins for histogram entropy estimation. Default is `10`.
- `base::Real`: Logarithmic base for entropy calculation. Default is `e` (natural logarithm).
- `verbose::Bool`: If `true`, print additional information. Default is `false`.
- `degenerate::Bool`: If `true`, adds noise to avoid extreme negative distances in degenerate cases. Default is `false`.
- `dim::Int`: If `1`, treat rows as samples and columns as variables (transpose the matrix). Default is `1`.

# Returns
- `Real`: The estimated entropy of the dataset.

# Example
```julia
entropy(rand(100), k=5)
entropy(rand(100), nbins=10, method="histogram")
```
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

"""
    entropy(array::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real

Estimate the entropy of a dataset using the k-nearest neighbors method.

# Arguments
- `array::Vector{<:Real}`: Input Vector containing the sample variables.
- `k::Int`: Number of neighbors for k-NN entropy estimation. Default is `3`.
- `base::Real`: Logarithmic base for entropy calculation. Default is `e` (natural logarithm).
- `verbose::Bool`: If `true`, print additional information. Default is `false`.
- `degenerate::Bool`: If `true`, adds noise to avoid extreme negative distances in degenerate cases. Default is `false`.
- `dim::Int`: If `1`, treat rows as samples and columns as variables (transpose the matrix). Default is `1`.

# Returns
- `Real`: The estimated entropy of the dataset.

# Example
```julia
entropy(rand(100), k=5)
```
"""
function entropy(array::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    mat_ = reshape(array, length(array), 1)
    return entropy(mat_, k=k, verbose=verbose, degenerate=degenerate, base=base, dim=dim, method=method, nbins=nbins)
end


"""
    conditional_entropy(X::Matrix{<:Real}, Y::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Compute the conditional entropy H(X|Y) using k-nearest neighbors.

# Arguments
- `X::Matrix{<:Real}`: The target matrix for which the conditional entropy is calculated.
- `Y::Matrix{<:Real}`: The conditioning matrix.
- `k::Int`: Number of neighbors for k-NN entropy estimation. Default is `3`.
- `base::Real`: Logarithmic base for entropy calculation. Default is `e` (natural logarithm).

# Returns
- `Real`: The estimated conditional entropy H(X|Y).

# Example
```julia
conditional_entropy(rand(100, 2), rand(100, 2), k=5)
```
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

"""
    conditional_entropy(array_::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Compute the conditional entropy H(X|Y) using k-nearest neighbors.

# Arguments
- `X::Vector{<:Real}`: The target matrix for which the conditional entropy is calculated.
- `Y::Vector{<:Real}`: The conditioning matrix.
- `k::Int`: Number of neighbors for k-NN entropy estimation. Default is `3`.
- `base::Real`: Logarithmic base for entropy calculation. Default is `e` (natural logarithm).

# Returns
- `Real`: The estimated conditional entropy H(X|Y).

# Example
```julia
conditional_entropy(rand(100), rand(100), k=5)
```
"""
function conditional_entropy(array_::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_ = reshape(array_, length(array_), 1)
    cond_ = reshape(cond_, length(cond_), 1)
    return conditional_entropy(array_, cond_, method=method, k=k, nbins=nbins, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    mutual_information(X::Matrix{<:Real}, Y::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, dim::Int = 1) -> Real

Calculate the mutual information I(X; Y) between two matrices X and Y using k-nearest neighbors.

# Arguments
- `X::Matrix{<:Real}`: The first matrix (set of variables).
- `Y::Matrix{<:Real}`: The second matrix (set of variables).
- `k::Int`: Number of neighbors for k-NN mutual information estimation. Default is `3`.
- `base::Real`: Logarithmic base for mutual information calculation. Default is `e` (natural logarithm).

# Returns
- `Real`: The estimated mutual information I(X; Y).

# Example
```julia
mutual_information(rand(100, 2), rand(100, 2), k=5)
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

"""
    mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the mutual information I(X; Y) between two vectors X and Y using k-nearest neighbors.

# Arguments
- `X::Matrix{<:Real}`: The first matrix (set of variables).
- `Y::Matrix{<:Real}`: The second matrix (set of variables).
- `k::Int`: Number of neighbors for k-NN mutual information estimation. Default is `3`.
- `base::Real`: Logarithmic base for mutual information calculation. Default is `e` (natural logarithm).

# Returns
- `Real`: The estimated mutual information I(X; Y).

# Example
```julia
mutual_information(rand(100), rand(100), k=5)
```
"""
function mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    mat_1 = reshape(array_1, length(array_1), 1)
    mat_2 = reshape(array_2, length(array_2), 1)
    return mutual_information(mat_1, mat_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end
                                
"""
    conditional_mutual_information(X::Matrix{<:Real}, Y::Matrix{<:Real}, Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the conditional mutual information (CMI) between X and Y, conditioned on Z.

The conditional mutual information I(X; Y | Z) quantifies the amount of information shared between X and Y, given the knowledge of Z. It helps in understanding dependencies between variables while controlling for a third variable.

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a sample and each row represents a variable from the first set.
- `Y::Matrix{<:Real}`: A matrix where each column represents a sample and each row represents a variable from the second set.
- `Z::Matrix{<:Real}`: A matrix representing the conditioning set of variables.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the mutual information calculation. Default is `e`.

# Returns
- `Real`: The estimated conditional mutual information I(X; Y | Z).

# Example
```julia
conditional_mutual_information(rand(100, 2), rand(100, 2), rand(100, 2), k=5)
```
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

"""
    conditional_mutual_information(X::Vector{<:Real}, Y::Vector{<:Real}, Z::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the conditional mutual information (CMI) between X and Y, conditioned on Z.

The conditional mutual information I(X; Y | Z) quantifies the amount of information shared between X and Y, given the knowledge of Z. It helps in understanding dependencies between variables while controlling for a third variable.

# Arguments
- `X::Vector{<:Real}`: A matrix where each column represents a sample and each row represents a variable from the first set.
- `Y::Vector{<:Real}`: A matrix where each column represents a sample and each row represents a variable from the second set.
- `Z::Vector{<:Real}`: A matrix representing the conditioning set of variables.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the mutual information calculation. Default is `e`.

# Returns
- `Real`: The estimated conditional mutual information I(X; Y | Z).

# Example
```julia
conditional_mutual_information(rand(100), rand(100), rand(100), k=5)
```
"""
function conditional_mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    cond_ = reshape(cond_, length(cond_), 1)
    return conditional_mutual_information(array_1, array_2, cond_, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end
                                                    
"""
    normalized_mutual_information(X::Matrix{<:Real}, Y::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the normalized mutual information (nMI) between two matrices X and Y using k-nearest neighbors.

The normalized mutual information (nMI) is a normalized version of mutual information, ensuring that the result lies between 0 and 1, where 0 indicates no shared information and 1 indicates perfect correlation.

# Arguments
- `X::Matrix{<:Real}`: The first matrix, with samples as columns and features as rows.
- `Y::Matrix{<:Real}`: The second matrix, structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the mutual information calculation. Default is `e`.

# Returns
- `Real`: The normalized mutual information, a value between 0 and 1.

# Example
```julia
nMI(rand(100, 2), rand(100, 2), k=5)
```
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

"""
    normalized_mutual_information(X::Vector{<:Real}, Y::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the normalized mutual information (nMI) between two vectors X and Y using k-nearest neighbors.

The normalized mutual information (nMI) is a normalized version of mutual information, ensuring that the result lies between 0 and 1, where 0 indicates no shared information and 1 indicates perfect correlation.

# Arguments
- `X::Vector{<:Real}`: The first matrix, with samples as columns and features as rows.
- `Y::Vector{<:Real}`: The second matrix, structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the mutual information calculation. Default is `e`.

# Returns
- `Real`: The normalized mutual information, a value between 0 and 1.

# Example
```julia
nMI(rand(100), rand(100), k=5)
```
"""
function normalized_mutual_information(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    return normalized_mutual_information(array_1, array_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end

"""
    interaction_information(X::Matrix{<:Real}; k::Int = 3, base::Real = e) -> Real

Calculate the interaction information (II) for a set of variables.

The interaction information measures the amount of redundancy or synergy among multiple variables. Positive interaction information suggests synergy (mutual dependence) among variables, while negative interaction information indicates redundancy (overlap of information).

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a sample and each row represents a variable.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The interaction information between the variables in `X`.

# Example
```julia
interaction_information(rand(100), rand(100), rand(100), k=5)
```
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

"""
    interaction_information(X::Vector{<:Real}, Y::Vector{<:Real}, Z::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the interaction information (II) for a set of variables.

The interaction information measures the amount of redundancy or synergy among multiple variables. Positive interaction information suggests synergy (mutual dependence) among variables, while negative interaction information indicates redundancy (overlap of information).

# Arguments
- `X::Vector{<:Real}`,`Y::Vector{<:Real}` and`Z::Vector{<:Real}` : The vectors containing the sample variables.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The interaction information between the variables in `X`.

# Example
```julia
interaction_information(rand(100), rand(100), rand(100), k=5)
```
"""
function interaction_information(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return interaction_information(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end
                                                                

"""
    redundancy(X::Matrix{<:Real}, Y::Matrix{<:Real}, Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the redundancy (Re) between two sets of variables X and Y about the variable Z.

Redundancy measures how much information is shared between two sets of variables. A higher redundancy indicates more overlap in the information provided by X and Y, meaning they share similar information content in Z.

# Arguments
- `X::Vector{<:Real}`,`Y::Vector{<:Real}` and`Z::Vector{<:Real}` : The vectors containing the sample variables.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The redundancy, representing the amount of shared information between `X` and `Y`about `Z`.

# Example
```julia
redundancy(rand(100), rand(100), rand(100), k=5)
```
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

"""
    redundancy(X::Matrix{<:Real}, Y::Matrix{<:Real}, Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the redundancy (Re) between two sets of variables X and Y about the variable Z.

Redundancy measures how much information is shared between two sets of variables. A higher redundancy indicates more overlap in the information provided by X and Y, meaning they share similar information content in Z.

# Arguments
- `X::Vector{<:Real}`,`Y::Vector{<:Real}` and`Z::Vector{<:Real}` : The vectors containing the sample variables.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The redundancy, representing the amount of shared information between `X` and `Y`about `Z`.

# Example
```julia
redundancy(rand(100), rand(100), rand(100), k=5)
```
"""
function redundancy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    array_3 = reshape(array_3, length(array_3), 1)
    return redundancy(array_1, array_2, array_3, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end


"""
    unique(X::Matrix{<:Real}, Y::Matrix{<:Real}, , Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> {Real, Real}

Calculate the unique information (Uni) between two sets of variables `X` and `Y` with respect to a target variable `Z`.

The unique information quantifies how much information is uniquely contributed by `X` and `Y` about `Z`, helping to distinguish individual contributions from shared information among variables.

# Arguments
- `X::Matrix{<:Real}`: A matrix where columns represent samples and rows represent features.
- `Y::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `Z::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `{Real, Real}`: A tuple containing two values:
  - The unique information in `X` about `Z`.
  - The unique information in `Y` about `Z`.

# Example
```julia
unique(rand(100), rand(100), rand(100), k=5)
```
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


"""
    unique(X::Matrix{<:Real}, Y::Matrix{<:Real}, , Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> {Real, Real}

Calculate the unique information (Uni) between two sets of variables `X` and `Y` with respect to a target variable `Z`.

The unique information quantifies how much information is uniquely contributed by `X` and `Y` about `Z`, helping to distinguish individual contributions from shared information among variables.

# Arguments
- `X::Matrix{<:Real}`: A matrix where columns represent samples and rows represent features.
- `Y::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `Z::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `{Real, Real}`: A tuple containing two values:
  - The unique information in `X` about `Z`.
  - The unique information in `Y` about `Z`.

# Example
```julia
unique(rand(100), rand(100), rand(100), k=5)
```
"""
function unique(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Tuple{Real, Real}
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return unique(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end


"""
    synergy(X::Matrix{<:Real}, Y::Matrix{<:Real}, , Z::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> {Real, Real}

Calculate the synergy (SYn) between three sets of variables X, Y and Z.

Synergy measures how much information is shared between three sets of variables.
# Arguments
- `X::Matrix{<:Real}`: A matrix where columns represent samples and rows represent features.
- `Y::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `Z::Matrix{<:Real}`: A matrix structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The synergy, representing the amount of shared information between `X`,  `Y` and `Z`.

# Example
```julia
synergy(rand(100), rand(100), rand(100), k=5)
```
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

"""
    synergy(X::Vector{<:Real}, Y::Vector{<:Real}, , Z::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Calculate the synergy (SYn) between three sets of variables X, Y and Z.

Synergy measures how much information is shared between three sets of variables.
# Arguments
- `X::Vector{<:Real}`: A matrix where columns represent samples and rows represent features.
- `Y::Vector{<:Real}`: A matrix structured similarly to `X`.
- `Z::Vector{<:Real}`: A matrix structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The synegy, representing the amount of shared information between `X`,  `Y` and `Z`.

# Example
```julia
synergy(rand(100), rand(100), rand(100), k=5)
```
"""
function synergy(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)        
    array_3 = reshape(array_3, length(array_3), 1)
    return synergy(array_1, array_2, array_3, method=method, nbins=nbins, k=k, base=base, verbose=verbose, degenerate=degenerate)
end

"""
    information_quality_ratio(X::Matrix{<:Real}, Y::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Compute the information quality ratio (IQR) between two variables X and Y.

The information quality ratio (IQR) assesses the usefulness of a variable for classification or prediction. It is often used in decision trees to evaluate the importance of a feature relative to others.

# Arguments
- `X::Matrix{<:Real}`: A matrix representing the feature set, with samples as columns and features as rows.
- `Y::Matrix{<:Real}`: A matrix representing the target variable(s), structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The information quality ratio, a measure of the utility of the variables in `X` for predicting `Y`.

# Example
```julia
information_quality_ratio(rand(100), rand(100), k=5)
```
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

"""
    information_quality_ratio(X::Vector{<:Real}, Y::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e) -> Real

Compute the information quality ratio (IQR) between two variables X and Y.

The information quality ratio (IQR) assesses the usefulness of a variable for classification or prediction. It is often used in decision trees to evaluate the importance of a feature relative to others.

# Arguments
- `X::Vector{<:Real}`: A matrix representing the feature set, with samples as columns and features as rows.
- `Y::Vector{<:Real}`: A matrix representing the target variable(s), structured similarly to `X`.
- `k::Int`: Number of neighbors for k-NN estimation. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Real`: The information quality ratio, a measure of the utility of the variables in `X` for predicting `Y`.

# Example
```julia
information_quality_ratio(rand(100), rand(100), k=5)
```
"""
function information_quality_ratio(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    array_1 = reshape(array_1, length(array_1), 1)
    array_2 = reshape(array_2, length(array_2), 1)
    return information_quality_ratio(array_1, array_2, method=method, nbins=nbins, k=k, verbose=verbose, degenerate=degenerate, base=base)
end



"""
    MI(mat_1::Matrix{<:Real}; k::Int = 3, base::Real = e) -> Matrix{Real}

Calculate the pairwise mutual information between each feature in the input matrix using k-nearest neighbors.

# Arguments
- `mat_1::Matrix{<:Real}`: A matrix where each column represents a feature, and each row represents a sample.
- `k::Int`: Number of neighbors for k-NN estimation of mutual information. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Matrix{Real}`: A square matrix containing the estimated mutual information between each pair of features.

# Example
```julia
MI(rand(1000, 5), k=5)
```
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
    CMI(mat_1::Matrix{<:Real}, array_1::array{<:Real}; k::Int = 3, base::Real = e) -> Matrix{Real}

Calculate the pairwise mutual information between each feature in the input matrix using k-nearest neighbors.

# Arguments
- `mat_1::Matrix{<:Real}`: A matrix where each column represents a feature, and each row represents a sample.
- `array_1::array{<:Real}`: An array representing the conditional variable.
- `k::Int`: Number of neighbors for k-NN estimation of mutual information. Default is `3`.
- `base::Real`: Logarithmic base for the calculation. Default is `e`.

# Returns
- `Matrix{Real}`: A square matrix containing the estimated conditional mutual information between each pair of features.

# Example
```julia
CMI(rand(1000, 5), rand(1000), k=5)
```
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
    if degenerate
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

"""
Type : ?entropy
"""
function ent(mat_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
   return entropy(mat_, k=k, verbose=verbose, degenerate=degenerate, base=base, dim=dim, method=method, nbins=nbins)
end
                
function ent(array::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return entropy(array, k=k, verbose=verbose, degenerate=degenerate, base=base, method=method, nbins=nbins) 
end

function nMI(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return normalized_mutual_information(mat_1, mat_2, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                                        
function nMI(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return normalized_mutual_information(array_1, array_2, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

"""
Type : ?mutual_information
"""
function MI(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return mutual_information(mat_1, mat_2, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                        
function MI(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return mutual_information(array_1, array_2, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

function CE(mat_::Matrix{<:Real}, cond_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return return conditional_entropy(mat_, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                
function CE(mat_::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return return conditional_entropy(mat_, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end
                        
function II(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return interaction_information(mat_1, mat_2, mat_3, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                
function II(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return interaction_information(array_1, array_2, array_3, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

function Uni(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Tuple{Real, Real}
    return unique(mat_1, mat_2, mat_3, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                
function Uni(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Tuple{Real, Real}
    return unique(array_1, array_2, array_3, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

function Re(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return redundancy(mat_1, mat_2, mat_3, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                            
function Re(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return redundancy(array_1, array_2, array_3, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end  

function Syn(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, mat_3::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return synergy(mat_1, mat_2, mat_3, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                            
function Syn(array_1::Vector{<:Real}, array_2::Vector{<:Real}, array_3::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return synergy(array_1, array_2, array_3, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end  

function IQR(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = 10, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return information_quality_ratio(mat_1, mat_2, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                                                                    
function IQR(array_1::Vector{<:Real}, array_2::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = 10, verbose::Bool = false, degenerate::Bool = false)::Real
    return information_quality_ratio(array_1, array_2, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

function CMI(mat_1::Matrix{<:Real}, mat_2::Matrix{<:Real}, cond_::Matrix{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false, dim::Int = 1)::Real
    return conditional_mutual_information(mat_1, mat_2, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, dim=dim, method=method, nbins=nbins)
end
                                                    
function CMI(array_1::Vector{<:Real}, array_2::Vector{<:Real}, cond_::Vector{<:Real};method::String = "inv", nbins::Int = 10, k::Int = 3, base::Real = e, verbose::Bool = false, degenerate::Bool = false)::Real
    return conditional_mutual_information(array_1, array_2, cond_, k=k, base=base, verbose=verbose, degenerate=degenerate, method=method, nbins=nbins)
end

end