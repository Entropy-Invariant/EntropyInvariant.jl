using EntropyInfo

println("Entropy")
println(EntropyInfo.entropy(rand(1000)))
println(EntropyInfo.entropy(rand(1, 1000), dim=2))
println(EntropyInfo.entropy(rand(1000, 1)))

println("Conditional entropy")
println(EntropyInfo.conditional_entropy(rand(1000), rand(1000)))
println(EntropyInfo.conditional_entropy(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.conditional_entropy(rand(1000, 1), rand(1000, 1)))

println("Mutual Information")
println(EntropyInfo.mutual_information(rand(1000), rand(1000)))
println(EntropyInfo.mutual_information(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.mutual_information(rand(1000, 1), rand(1000, 1)))

println("Normalized Mutual Information")
println(EntropyInfo.normalized_mutual_information(rand(1000), rand(1000)))
println(EntropyInfo.normalized_mutual_information(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.normalized_mutual_information(rand(1000, 1), rand(1000, 1)))

println("Conditional Mutual Information")
println(EntropyInfo.conditional_mutual_information(rand(1000), rand(1000), rand(1000)))
println(EntropyInfo.conditional_mutual_information(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.conditional_mutual_information(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Unique")
println(EntropyInfo.unique(rand(1000), rand(1000), rand(1000)))
println(EntropyInfo.unique(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.unique(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Redundancy")
println(EntropyInfo.redundancy(rand(1000), rand(1000), rand(1000)))
println(EntropyInfo.redundancy(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.redundancy(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Synergy")
println(EntropyInfo.synergy(rand(1000), rand(1000), rand(1000)))
println(EntropyInfo.synergy(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.synergy(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Information Quality Ratio")
println(EntropyInfo.information_quality_ratio(rand(1000), rand(1000)))
println(EntropyInfo.information_quality_ratio(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInfo.information_quality_ratio(rand(1000, 1), rand(1000, 1)))

"""
println("2D - Mutual Information")
println(EntropyInfo.mutual_information(rand(1000, 5)))
println(EntropyInfo.mutual_information(rand(5, 1000), dim=2))

println("2D - COnditional Mutual Information")
println(EntropyInfo.conditional_mutual_information(rand(1000, 5), rand(1000)))
println(EntropyInfo.conditional_mutual_information(rand(5, 1000), rand(1, 1000), dim=2))
"""