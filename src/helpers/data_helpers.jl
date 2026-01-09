# Data layout and validation helper functions for EntropyInvariant package

"""
    get_shape(mat::Matrix{<:Real}) -> DataShape

Extract shape information from a matrix in canonical format (dim=2).

In canonical format:
- Each column is a data point
- Each row is a dimension

# Arguments
- `mat::Matrix{<:Real}`: Data matrix in canonical (dim=2) format

# Returns
- `DataShape`: Struct containing num_points and num_dimensions

# Example
```julia
mat = rand(3, 100)  # 3 dimensions, 100 points
shape = get_shape(mat)
# shape.num_dimensions == 3
# shape.num_points == 100
```
"""
function get_shape(mat::Matrix{<:Real})::DataShape
    return DataShape(size(mat, 2), size(mat, 1))
end

"""
    ensure_columns_are_points(mat::Matrix{<:Real}, dim::Int) -> Matrix{Float64}

Convert matrix to canonical layout where each column is a data point.

The package uses a specific data layout convention:
- `dim=1` (default): Input has data points as rows, dimensions as columns → transpose needed
- `dim=2`: Input has data points as columns, dimensions as rows → no change needed

This function eliminates 12 instances of duplicate transpose logic throughout the codebase.

# Arguments
- `mat::Matrix{<:Real}`: Input data matrix
- `dim::Int`: Layout indicator (1 or 2)

# Returns
- `Matrix{Float64}`: Matrix in canonical format (columns as points)

# Example
```julia
# Input with points as rows (dim=1)
mat_rows = rand(100, 3)  # 100 points, 3 dimensions
canonical = ensure_columns_are_points(mat_rows, 1)
# Result: 3×100 matrix (transposed)

# Input with points as columns (dim=2)
mat_cols = rand(3, 100)  # 3 dimensions, 100 points
canonical = ensure_columns_are_points(mat_cols, 2)
# Result: 3×100 matrix (unchanged)
```
"""
function ensure_columns_are_points(mat::Matrix{<:Real}, dim::Int)::Matrix{Float64}
    return dim == 1 ? Matrix{Float64}(transpose(mat)) : Matrix{Float64}(mat)
end

"""
    vector_to_matrix(vec::Vector{<:Real}) -> Matrix{Float64}

Convert a 1D vector to an n×1 matrix for consistent processing.

This function eliminates 24 instances of `reshape(array, length(array), 1)`
throughout the codebase.

# Arguments
- `vec::Vector{<:Real}`: Input vector

# Returns
- `Matrix{Float64}`: Column matrix with same data

# Example
```julia
x = [1.0, 2.0, 3.0]
mat = vector_to_matrix(x)
# Result: 3×1 matrix
```
"""
function vector_to_matrix(vec::Vector{<:Real})::Matrix{Float64}
    return reshape(vec, length(vec), 1)
end

"""
    validate_same_num_points(shapes::Vector{DataShape})

Validate that all datasets have the same number of data points.

Eliminates 9 instances of duplicate validation logic throughout the codebase.

# Arguments
- `shapes::Vector{DataShape}`: Vector of DataShape objects to validate

# Throws
- `ArgumentError`: If datasets have different numbers of points

# Example
```julia
shape1 = DataShape(100, 2)
shape2 = DataShape(100, 3)
validate_same_num_points([shape1, shape2])  # OK

shape3 = DataShape(50, 2)
validate_same_num_points([shape1, shape3])  # Throws ArgumentError
```
"""
function validate_same_num_points(shapes::Vector{DataShape})
    if length(shapes) < 2
        return  # Nothing to validate
    end

    first_num_points = shapes[1].num_points
    for shape in shapes[2:end]
        if shape.num_points != first_num_points
            throw(ArgumentError("Input arrays must contain the same number of points"))
        end
    end
end

"""
    validate_dimensions_equal_one(shapes::Vector{DataShape})

Validate that all datasets are 1-dimensional.

# Arguments
- `shapes::Vector{DataShape}`: Vector of DataShape objects to validate

# Throws
- `ArgumentError`: If any dataset has num_dimensions != 1

# Example
```julia
shape1 = DataShape(100, 1)
shape2 = DataShape(100, 1)
validate_dimensions_equal_one([shape1, shape2])  # OK

shape3 = DataShape(100, 2)
validate_dimensions_equal_one([shape1, shape3])  # Throws ArgumentError
```
"""
function validate_dimensions_equal_one(shapes::Vector{DataShape})
    for shape in shapes
        if shape.num_dimensions != 1
            throw(ArgumentError("Each input must be 1-dimensional"))
        end
    end
end
