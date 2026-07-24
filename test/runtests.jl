using EntropyInvariant
using Test
using Random

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

@testset "Matrix/scalar consistency on sparse data" begin
    # Regression tests for a bug where MI()/CMI() (matrix fast-path) computed
    # the invariant measure inline, without EntropyInvariant's own
    # zero-filtering (compute_invariant_measure filters exact zeros before
    # taking the median nearest-neighbor distance -- sparse data, common in
    # real signals like mass-spec bins, is mostly zeros). This silently
    # diverged from -- and eventually crashed relative to -- the scalar
    # mutual_information()/conditional_mutual_information() functions, which
    # always went through the correct, zero-filtered helper. A second,
    # separate bug (digamma(n) using the post-zero-filtering distance count
    # instead of the true sample size) compounded this for method="inv".

    make_sparse_column(n) = begin
        col = zeros(n)
        nonzero_idx = randperm(n)[1:(n ÷ 5)]
        col[nonzero_idx] = rand(length(nonzero_idx)) .* 10 .+ 1.0
        col
    end

    n = 500
    x = make_sparse_column(n)
    y = make_sparse_column(n)
    z = rand(n) .* 10 .+ 1.0  # dense conditioning variable, no zeros -- see
                              # "Degenerate KSG radius" testset below for why

    for method in ("inv", "inv_ksg")
        data = hcat(x, y)
        cmi_mat = EntropyInvariant.CMI(data, z, method=method, k=5)
        cmi_direct = conditional_mutual_information(x, y, z, method=method, k=5)
        @test all(isfinite, cmi_mat)
        @test abs(cmi_mat[1, 2] - cmi_direct) < 1e-9
    end

    # method="inv": no shared radius involved, so heavy (~80%) sparsity on
    # both x and y (no conditioning z to break ties) is fine here.
    mi_mat_inv = EntropyInvariant.MI(hcat(x, y), method="inv", k=5)
    mi_direct_inv = mutual_information(x, y, method="inv", k=5)
    @test all(isfinite, mi_mat_inv)
    @test abs(mi_mat_inv[1, 2] - mi_direct_inv) < 1e-9

    # method="inv_ksg": with no conditioning z, x and y both being heavily
    # sparse would make the shared KSG radius legitimately degenerate (see
    # "Degenerate KSG radius" testset) -- that's correct behavior, not a
    # bug, so this uses much lighter (~2%) sparsity instead.
    make_mildly_sparse_column(n) = begin
        col = zeros(n)
        nonzero_idx = randperm(n)[1:round(Int, n * 0.98)]
        col[nonzero_idx] = rand(length(nonzero_idx)) .* 10 .+ 1.0
        col
    end
    mx = make_mildly_sparse_column(n)
    my = make_mildly_sparse_column(n)
    mi_mat_ksg = EntropyInvariant.MI(hcat(mx, my), method="inv_ksg", k=5)
    mi_direct_ksg = mutual_information(mx, my, method="inv_ksg", k=5)
    @test all(isfinite, mi_mat_ksg)
    @test abs(mi_mat_ksg[1, 2] - mi_direct_ksg) < 1e-9
end

@testset "Degenerate invariant measure fails loudly" begin
    # A column where >=half the non-zero values are exact duplicates, so the
    # median nearest-neighbor distance is exactly 0.
    duplicate_heavy = Float64.(rand(1:3, 200))
    x = rand(200)

    @test_throws ArgumentError EntropyInvariant.compute_invariant_measure(duplicate_heavy)
    @test_throws ArgumentError EntropyInvariant.MI(hcat(duplicate_heavy, x))
    @test_throws ArgumentError EntropyInvariant.CMI(hcat(duplicate_heavy, x), rand(200))
end

@testset "Degenerate KSG radius fails loudly" begin
    # x, y both ~80% zero at independently-chosen positions: by the
    # pigeonhole principle, a large block of points must be (0, 0) exactly,
    # making the shared KSG radius (and thus digamma(0) = -Inf) degenerate.
    n = 500
    make_col() = begin
        col = zeros(n)
        nonzero_idx = randperm(n)[1:(n ÷ 5)]
        col[nonzero_idx] = rand(length(nonzero_idx)) .* 10 .+ 1.0
        col
    end
    x = make_col()
    y = make_col()

    @test_throws ArgumentError mutual_information_ksg(x, y, k=5)
    @test_throws ArgumentError EntropyInvariant.MI(hcat(x, y), method="inv_ksg", k=5)

    # z also sparse here (unlike the dense z above), so the full (x, y, z)
    # joint space is degenerate too.
    z = make_col()
    @test_throws ArgumentError EntropyInvariant.CMI(hcat(x, y), z, method="inv_ksg", k=5)
end

@testset "Self-MI diagonal special case" begin
    # I(X;X) = H(X): pairing a sparse/duplicate-heavy variable with itself
    # should not hit the degenerate-radius error the previous testset
    # exercises deliberately -- MI()'s diagonal and mutual_information_ksg(x,
    # x) both special-case this instead of running the shared-radius trick.
    n = 500
    x = zeros(n)
    nonzero_idx = randperm(n)[1:(n ÷ 5)]
    x[nonzero_idx] = rand(length(nonzero_idx)) .* 10 .+ 1.0

    self_mi = mutual_information_ksg(x, x, k=5)
    @test isfinite(self_mi)

    mi_mat = EntropyInvariant.MI(hcat(x, rand(n)), method="inv_ksg", k=5)
    @test isfinite(mi_mat[1, 1])
    @test abs(mi_mat[1, 1] - self_mi) < 1e-9
end

@testset "parallel=true matches sequential" begin
    # Threads.@threads over pairs instead of requiring users to hand-roll
    # parallelism (as we did ad hoc for the JASM analysis this generalizes).
    # Result should be identical to parallel=false regardless of how many
    # threads Julia was started with (falls back to sequential-on-one-thread
    # if only one is available, per Threads.@threads semantics).
    n = 300
    data = rand(n, 6)
    z = rand(n)

    mi_seq = EntropyInvariant.MI(data, method="inv_ksg", k=4, parallel=false)
    mi_par = EntropyInvariant.MI(data, method="inv_ksg", k=4, parallel=true)
    @test mi_seq ≈ mi_par

    cmi_seq = EntropyInvariant.CMI(data, z, method="inv_ksg", k=4, parallel=false)
    cmi_par = EntropyInvariant.CMI(data, z, method="inv_ksg", k=4, parallel=true)
    @test cmi_seq ≈ cmi_par
end
