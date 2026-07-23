using EntropyInvariant
using Test

@testset "EntropyInvariant.jl" begin
    # Dimensionality and consistency tests
    # Test for Entropy function
    n = 1000
    x = rand(n)
    actual_output = entropy(x)
    expected_output1 = entropy(reshape(x, n, 1))
    expected_output2 = entropy(reshape(x, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Mutual Information
    y = rand(n)
    actual_output = mutual_information(x, y)
    expected_output1 = mutual_information(reshape(x, n, 1), reshape(y, n, 1))
    expected_output2 = mutual_information(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7

    # Test for Conditional Entropy
    actual_output = conditional_entropy(x,y)
    expected_output1 = conditional_entropy(reshape(x, n, 1), reshape(y, n, 1))
    expected_output2 = conditional_entropy(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Joint Entropy
    x = rand(n,2)
    actual_output = entropy(x)
    expected_output1 = entropy(Matrix(transpose(x)), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    
    # Test for Conditional Mutual Information
    x = rand(n)
    y = rand(n)
    z = rand(n)
    actual_output = conditional_mutual_information(x,y,z)
    expected_output1 = conditional_mutual_information(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = conditional_mutual_information(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Interaction Information
    actual_output = interaction_information(x,y,z)
    expected_output1 = interaction_information(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = interaction_information(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Redundancy
    actual_output = redundancy(x,y,z) 
    expected_output1 = redundancy(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = redundancy(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Normalized Mutual Information
    actual_output = normalized_mutual_information(x,y)
    expected_output1 = normalized_mutual_information(reshape(x, n, 1),reshape(y, n, 1))
    expected_output2 = normalized_mutual_information(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    
    # Test for Information Quality Ratio
    actual_output = information_quality_ratio(x,y)
    expected_output1 = information_quality_ratio(reshape(x, n, 1),reshape(y, n, 1))
    expected_output2 = information_quality_ratio(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7

    # Test for Unique (qualified to avoid conflict with Base.unique)
    actual_output = EntropyInvariant.unique(x,y,z)
    expected_output1 = EntropyInvariant.unique(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = EntropyInvariant.unique(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output[1]-expected_output1[1]) < 1e-7
    @test abs(actual_output[2]-expected_output1[2]) < 1e-7
    @test abs(actual_output[1]-expected_output2[1]) < 1e-7
    @test abs(actual_output[2]-expected_output2[2]) < 1e-7

    # Test for Synergy
    actual_output = synergy(x,y,z)
    expected_output1 = synergy(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = synergy(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7

    # Test for Optimized Mutual Information Matrix (MI function)
    m = 3
    a = rand(n, m)
    actual_output = zeros(m, m)
    for i in 1:m
        for j in 1:m
            actual_output[i,j] = mutual_information(a[:,i], a[:,j])
        end
    end
    expected_output1 = EntropyInvariant.MI(a)
    expected_output2 = EntropyInvariant.MI(Matrix(transpose(a)), dim=2)
    @test maximum(abs.(actual_output - expected_output1)) < 1e-7
    @test maximum(abs.(actual_output - expected_output2)) < 1e-7

    # Test for Optimized Conditional Mutual Information Matrix (CMI function)
    b = rand(n)
    actual_output = zeros(m, m)
    for i in 1:m
        for j in 1:m
            actual_output[i,j] = conditional_mutual_information(a[:,i], a[:,j], b)
        end
    end
    expected_output1 = EntropyInvariant.CMI(a, b)
    expected_output2 = EntropyInvariant.CMI(a, reshape(b, n, 1))
    expected_output3 = EntropyInvariant.CMI(Matrix(transpose(a)), b, dim=2)
    expected_output4 = EntropyInvariant.CMI(Matrix(transpose(a)), reshape(b, 1, n), dim=2)
    @test maximum(abs.(actual_output - expected_output1)) < 1e-7
    @test maximum(abs.(actual_output - expected_output2)) < 1e-7
    @test maximum(abs.(actual_output - expected_output3)) < 1e-7
    @test maximum(abs.(actual_output - expected_output4)) < 1e-7
end

@testset "KSG / Frenzel-Pompe (inv_ksg)" begin
    n = 2000
    x = rand(n)
    y = 2 * x .+ 0.1 * rand(n)
    z = rand(n)

    # --- mutual_information_ksg ---
    mi = mutual_information_ksg(x, y)
    @test isfinite(mi)
    @test abs(mutual_information_ksg(x, y) - mutual_information_ksg(y, x)) < 1e-10

    # Scale invariance: independent affine rescaling of each variable
    mi_scaled = mutual_information_ksg(1e6 * x .- 5, 1e-6 * y .+ 3)
    @test abs(mi - mi_scaled) < 1e-6

    # Closed-form check: bivariate Gaussian, I(X;Y) = -0.5*log(1-rho^2)
    rho = 0.6
    z1 = randn(5000); z2 = randn(5000)
    gx = z1
    gy = rho .* z1 .+ sqrt(1 - rho^2) .* z2
    true_mi = -0.5 * log(1 - rho^2)
    @test abs(mutual_information_ksg(gx, gy) - true_mi) < 0.05

    # Independent variables: MI close to 0
    ind_x = randn(3000); ind_y = randn(3000)
    @test abs(mutual_information_ksg(ind_x, ind_y)) < 0.1

    # --- conditional_mutual_information_ksg ---
    cmi = conditional_mutual_information_ksg(x, y, z)
    @test isfinite(cmi)

    cmi_scaled = conditional_mutual_information_ksg(1e6 * x .- 5, 1e-6 * y, 1e3 * z)
    @test abs(cmi - cmi_scaled) < 1e-6

    # Chain X -> Z -> Y: I(X;Y|Z) should be close to 0
    cx = randn(3000)
    cz = cx .+ 0.3 * randn(3000)
    cy = cz .+ 0.3 * randn(3000)
    @test abs(conditional_mutual_information_ksg(cx, cy, cz)) < 0.1

    # Collider X -> Z <- Y: I(X;Y|Z) should be clearly positive
    colx = randn(3000); coly = randn(3000)
    colz = colx .+ coly .+ 0.3 * randn(3000)
    @test conditional_mutual_information_ksg(colx, coly, colz) > 0.3

    # --- inv_ksg is the default for every MI/CMI-derived quantity ---
    @test mutual_information(x, y) == mutual_information_ksg(x, y)
    @test conditional_mutual_information(x, y, z) == conditional_mutual_information_ksg(x, y, z)

    ent_y = entropy(y, method="inv")
    @test abs(conditional_entropy(x, y) - (ent_y - mutual_information_ksg(x, y))) < 1e-10

    mi_12 = mutual_information_ksg(x, z)
    cmi_123 = conditional_mutual_information_ksg(x, y, z)
    @test abs(interaction_information(x, y, z) - (mutual_information_ksg(x, y) - cmi_123)) < 1e-10

    r = redundancy(x, y, z)
    expected_r = min(mutual_information_ksg(x, z), mutual_information_ksg(y, z))
    @test abs(r - expected_r) < 1e-10

    # NMI / IQR: finite and scale-invariant
    nmi = normalized_mutual_information(x, y)
    nmi_scaled = normalized_mutual_information(1e6 * x .- 5, 1e-6 * y .+ 3)
    @test abs(nmi - nmi_scaled) < 1e-5

    iqr = information_quality_ratio(x, y)
    iqr_scaled = information_quality_ratio(1e6 * x .- 5, 1e-6 * y .+ 3)
    @test abs(iqr - iqr_scaled) < 1e-5

    # --- MI / CMI matrices default to inv_ksg, "inv" still available ---
    data = hcat(x, y, z)
    mi_mat = EntropyInvariant.MI(data)
    @test abs(mi_mat[1, 2] - mutual_information_ksg(x, y)) < 1e-10
    @test mi_mat ≈ mi_mat'
    @test all(isfinite, mi_mat)

    mi_mat_inv = EntropyInvariant.MI(data, method="inv")
    @test all(isfinite, mi_mat_inv)
    @test_throws ArgumentError EntropyInvariant.MI(data, method="bogus")

    cmi_mat = EntropyInvariant.CMI(hcat(x, y), z)
    @test abs(cmi_mat[1, 2] - conditional_mutual_information_ksg(x, y, z)) < 1e-10
    @test cmi_mat ≈ cmi_mat'

    cmi_mat_inv = EntropyInvariant.CMI(hcat(x, y), z, method="inv")
    @test all(isfinite, cmi_mat_inv)
end
