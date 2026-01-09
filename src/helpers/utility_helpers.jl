# Utility helper functions for EntropyInvariant package

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
- Each data point is assigned to a bin if it falls within the range of that bin. The bin edges are inclusive on the lower end and exclusive on the upper end, except for the last bin, which is inclusive on both ends to ensure all data points are accounted for.
"""
function hist1d(data, nbins)
    min_val, max_val = minimum(data), maximum(data)
    bin_edges = range(min_val, stop=max_val, length=nbins+1)
    counts = zeros(Int, nbins)
    for value in data
        for i in 1:nbins
            # Last bin is inclusive on both ends: [edge_i, edge_{i+1}]
            # Other bins are left-inclusive, right-exclusive: [edge_i, edge_{i+1})
            if i == nbins
                if value >= bin_edges[i] && value <= bin_edges[i+1]
                    counts[i] += 1
                    break
                end
            else
                if value >= bin_edges[i] && value < bin_edges[i+1]
                    counts[i] += 1
                    break
                end
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
    for k in eachindex(x)
        # Find the appropriate bin index for x[k] and y[k]
        i = findfirst(t -> t >= x[k], bin_edges_x) - 1
        j = findfirst(t -> t >= y[k], bin_edges_y) - 1

        # Handle max values (last bin is inclusive on upper end)
        if x[k] == max_x
            i = nbins
        end
        if y[k] == max_y
            j = nbins
        end

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
    for k in eachindex(x)
        # Find the appropriate bin index for x[k], y[k], and z[k]
        i = findfirst(t -> t >= x[k], bin_edges_x) - 1
        j = findfirst(t -> t >= y[k], bin_edges_y) - 1
        l = findfirst(t -> t >= z[k], bin_edges_z) - 1

        # Handle max values (last bin is inclusive on upper end)
        if x[k] == max_x
            i = nbins
        end
        if y[k] == max_y
            j = nbins
        end
        if z[k] == max_z
            l = nbins
        end

        # Ensure the index is within the bounds of the counts array
        if 1 <= i <= nbins && 1 <= j <= nbins && 1 <= l <= nbins
            counts[i, j, l] += 1
        end
    end
    return counts
end

"""
    log_computation_info(shape::DataShape, base::Real)

Print computation information for a single dataset.

Eliminates 14 instances of duplicate verbose logging throughout the codebase.

# Arguments
- `shape::DataShape`: Dataset shape information
- `base::Real`: Logarithmic base being used

# Example
```julia
shape = DataShape(1000, 3)
log_computation_info(shape, 2)
# Prints:
# Number of points: 1000
# Dimensions: 3
# Base: 2
```
"""
function log_computation_info(shape::DataShape, base::Real)
    println("Number of points: $(shape.num_points)")
    println("Dimensions: $(shape.num_dimensions)")
    println("Base: $base")
end

"""
    log_computation_info(shapes::Vector{DataShape}, base::Real)

Print computation information for multiple datasets (e.g., joint computation).

# Arguments
- `shapes::Vector{DataShape}`: Vector of dataset shapes
- `base::Real`: Logarithmic base being used

# Example
```julia
shape1 = DataShape(1000, 1)
shape2 = DataShape(1000, 1)
log_computation_info([shape1, shape2], e)
# Prints:
# Number of points: 1000
# Dimensions: 2  (total across all datasets)
# Base: 2.718...
```
"""
function log_computation_info(shapes::Vector{DataShape}, base::Real)
    total_dims = sum(s.num_dimensions for s in shapes)
    println("Number of points: $(shapes[1].num_points)")
    println("Dimensions: $total_dims")
    println("Base: $base")
end
