# Type definitions for EntropyInvariant package

"""
    DataShape

A struct to store shape information for datasets.

# Fields
- `num_points::Int`: Number of data points
- `num_dimensions::Int`: Number of dimensions
"""
struct DataShape
    num_points::Int
    num_dimensions::Int
end

"""
    KNNResult

A struct to store results from k-nearest neighbor computations.

# Fields
- `indices::Vector{Vector{Int}}`: Indices of k-nearest neighbors for each point
- `all_distances::Vector{Vector{Float64}}`: All k-nearest neighbor distances for each point
- `kth_distances::Vector{Float64}`: Distance to the k-th nearest neighbor for each point
"""
struct KNNResult
    indices::Vector{Vector{Int}}
    all_distances::Vector{Vector{Float64}}
    kth_distances::Vector{Float64}
end
