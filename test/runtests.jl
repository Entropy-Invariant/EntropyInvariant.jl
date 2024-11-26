using EntropyInfo
using Test

@testset "EntropyInfo.jl" begin
    # Dimensionality and consistency tests
    # Test for Entropy function
    n = 1000
    x = rand(n)
    actual_output = entropy(x)
    expected_output1 = entropy(reshape(x, n, 1))
    expected_output2 = entropy(reshape(x, 1, n), dim=2)
    expected_output3 = EntropyInfo.ent(x)
    expected_output4 = EntropyInfo.ent(reshape(x, n, 1))
    expected_output5 = EntropyInfo.ent(reshape(x, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Mutual Information
    y = rand(n)
    actual_output = mutual_information(x, y)
    expected_output1 = mutual_information(reshape(x, n, 1), reshape(y, n, 1))
    expected_output2 = mutual_information(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    expected_output3 = EntropyInfo.MI(x, y)
    expected_output4 = EntropyInfo.MI(reshape(x, n, 1), reshape(y, n, 1))
    expected_output5 = EntropyInfo.MI(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    expected_output6 = EntropyInfo.ent(x)+EntropyInfo.ent(y)-EntropyInfo.ent(hcat(x,y))
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    @test abs(actual_output-expected_output6) < 1e-7

    # Test for Conditional Entropy
    actual_output = conditional_entropy(x,y)
    expected_output1 = conditional_entropy(reshape(x, n, 1), reshape(y, n, 1))
    expected_output2 = conditional_entropy(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    expected_output3 = EntropyInfo.CE(x,y)
    expected_output4 = EntropyInfo.CE(reshape(x, n, 1), reshape(y, n, 1))
    expected_output5 = EntropyInfo.CE(reshape(x, 1, n), reshape(y, 1, n), dim=2)
    expected_output6 = EntropyInfo.ent(hcat(x,y))-EntropyInfo.ent(x)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    @test abs(actual_output-expected_output6) < 1e-7
    
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
    expected_output3 = EntropyInfo.CMI(x,y,z)
    expected_output4 = EntropyInfo.CMI(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output5 = EntropyInfo.CMI(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Interaction Information
    actual_output = interaction_information(x,y,z)
    expected_output1 = interaction_information(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = interaction_information(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    expected_output3 = EntropyInfo.II(x,y,z)
    expected_output4 = EntropyInfo.II(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output5 = EntropyInfo.II(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Redundancy
    actual_output = redundancy(x,y,z) 
    expected_output1 = redundancy(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output2 = redundancy(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    expected_output3 = EntropyInfo.Re(x,y,z)
    expected_output4 = EntropyInfo.Re(reshape(x, n, 1),reshape(y, n, 1),reshape(z, n, 1))
    expected_output5 = EntropyInfo.Re(reshape(x, 1, n),reshape(y, 1, n),reshape(z, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Normalized Mutual Information
    actual_output = normalized_mutual_information(x,y)
    expected_output1 = normalized_mutual_information(reshape(x, n, 1),reshape(y, n, 1))
    expected_output2 = normalized_mutual_information(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    expected_output3 = EntropyInfo.nMI(x,y)
    expected_output4 = EntropyInfo.nMI(reshape(x, n, 1),reshape(y, n, 1))
    expected_output5 = EntropyInfo.nMI(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Information Quality Ratio
    actual_output = information_quality_ratio(x,y)
    expected_output1 = information_quality_ratio(reshape(x, n, 1),reshape(y, n, 1))
    expected_output2 = information_quality_ratio(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    expected_output3 = EntropyInfo.IQR(x,y)
    expected_output4 = EntropyInfo.IQR(reshape(x, n, 1),reshape(y, n, 1))
    expected_output5 = EntropyInfo.IQR(reshape(x, 1, n),reshape(y, 1, n), dim=2)
    @test abs(actual_output-expected_output1) < 1e-7
    @test abs(actual_output-expected_output2) < 1e-7
    @test abs(actual_output-expected_output3) < 1e-7
    @test abs(actual_output-expected_output4) < 1e-7
    @test abs(actual_output-expected_output5) < 1e-7
    
    # Test for Mutual Information Matrix
    m = 3
    a = rand(n, 3)
    actual_output = zeros(m,m)
    for i in 1:m
        for j in 1:m
            actual_output[i,j] = EntropyInfo.MI(a[:,i], a[:,j])
        end
    end
    expected_output1 = EntropyInfo.MI(a)
    expected_output2 = EntropyInfo.MI(Matrix(transpose(a)), dim=2)
    @test maximum(abs.(actual_output-expected_output1)) < 1e-7
    @test maximum(abs.(actual_output-expected_output2)) < 1e-7
    
    # Test for Conditional Mutual Information Matrix
    b = rand(n)
    actual_output = zeros(m,m)
    for i in 1:m
        for j in 1:m
            actual_output[i,j] = EntropyInfo.CMI(a[:,i], a[:,j], b)
        end
    end
    expected_output1 = EntropyInfo.CMI(a, b)
    expected_output2 = EntropyInfo.CMI(a, reshape(b, n, 1))
    expected_output3 = EntropyInfo.CMI(Matrix(transpose(a)), b, dim=2)
    expected_output4 = EntropyInfo.CMI(Matrix(transpose(a)), reshape(b, 1, n), dim=2)
    @test maximum(abs.(actual_output-expected_output1)) < 1e-7
    @test maximum(abs.(actual_output-expected_output2)) < 1e-7
    @test maximum(abs.(actual_output-expected_output3)) < 1e-7
    @test maximum(abs.(actual_output-expected_output4)) < 1e-7
end