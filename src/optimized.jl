# Optimized Matrix Functions for Information Theory Computation
#
# This file contains optimized implementations for computing information-theoretic
# quantities on matrices. These functions provide faster computation for specific
# cases like 2D pairwise mutual information.
#
# Functions included:
# - MI(a::Matrix): Compute pairwise mutual information for all dimension pairs
# - CMI(a::Matrix, b::Vector): Compute pairwise conditional MI matrix given conditioning variable
#
# These optimized functions are used internally for performance improvements

"""
    MI(X::Matrix{<:Real}; k::Int = 3, base::Real = e, verbose::Bool = false,
       degenerate::Bool = false, dim::Int = 1) -> Matrix{<:Real}

Compute the pairwise mutual information (MI) matrix for all pairs of dimensions in a dataset,
using the k-nearest neighbors (k-NN) invariant measure.

For each pair of dimensions (i, j), computes: MI(Xᵢ; Xⱼ) = H(Xᵢ) + H(Xⱼ) - H(Xᵢ, Xⱼ)

# Arguments
- `X::Matrix{<:Real}`: A matrix where each column represents a dimension of the dataset.
- `k::Int = 3` (optional): The number of nearest neighbors for k-NN estimation. Defaults to 3.
- `base::Real = e` (optional): The logarithmic base for MI computation. Defaults to natural logarithm (`e`).
- `verbose::Bool = false` (optional): If `true`, prints information about the dataset. Defaults to `false`.
- `degenerate::Bool = false` (optional): If `true`, adds noise to distances for degenerate cases. Defaults to `false`.
- `dim::Int = 1` (optional): Data layout: `dim=1` for rows as points (default), `dim=2` for columns as points.

# Returns
- `Matrix{<:Real}`: A symmetric matrix M where M[i,j] is the mutual information between dimensions i and j.

# Example
```julia
# Compute pairwise MI for a 5-dimensional dataset
data = rand(1000, 5)  # 1000 points in 5 dimensions
mi_matrix = MI(data)

# MI with custom parameters
mi_matrix = MI(data, k=5, base=2, verbose=true)

# For column-oriented data
data_cols = rand(5, 1000)  # 5 dimensions × 1000 points
mi_matrix = MI(data_cols, dim=2)
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

