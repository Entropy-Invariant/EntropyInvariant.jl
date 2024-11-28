using EntropyInvariant

println("Entropy")
println(EntropyInvariant.entropy(rand(1000)))
println(EntropyInvariant.entropy(rand(1, 1000), dim=2))
println(EntropyInvariant.entropy(rand(1000, 1)))

println("Conditional entropy")
println(EntropyInvariant.conditional_entropy(rand(1000), rand(1000)))
println(EntropyInvariant.conditional_entropy(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.conditional_entropy(rand(1000, 1), rand(1000, 1)))

println("Mutual Information")
println(EntropyInvariant.mutual_information(rand(1000), rand(1000)))
println(EntropyInvariant.mutual_information(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.mutual_information(rand(1000, 1), rand(1000, 1)))

println("Normalized Mutual Information")
println(EntropyInvariant.normalized_mutual_information(rand(1000), rand(1000)))
println(EntropyInvariant.normalized_mutual_information(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.normalized_mutual_information(rand(1000, 1), rand(1000, 1)))

println("Conditional Mutual Information")
println(EntropyInvariant.conditional_mutual_information(rand(1000), rand(1000), rand(1000)))
println(EntropyInvariant.conditional_mutual_information(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.conditional_mutual_information(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Unique")
println(EntropyInvariant.unique(rand(1000), rand(1000), rand(1000)))
println(EntropyInvariant.unique(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.unique(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Redundancy")
println(EntropyInvariant.redundancy(rand(1000), rand(1000), rand(1000)))
println(EntropyInvariant.redundancy(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.redundancy(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Synergy")
println(EntropyInvariant.synergy(rand(1000), rand(1000), rand(1000)))
println(EntropyInvariant.synergy(rand(1, 1000), rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.synergy(rand(1000, 1), rand(1000, 1), rand(1000, 1)))

println("Information Quality Ratio")
println(EntropyInvariant.information_quality_ratio(rand(1000), rand(1000)))
println(EntropyInvariant.information_quality_ratio(rand(1, 1000), rand(1, 1000), dim=2))
println(EntropyInvariant.information_quality_ratio(rand(1000, 1), rand(1000, 1)))

"""
println("2D - Mutual Information")
println(EntropyInvariant.mutual_information(rand(1000, 5)))
println(EntropyInvariant.mutual_information(rand(5, 1000), dim=2))

println("2D - COnditional Mutual Information")
println(EntropyInvariant.conditional_mutual_information(rand(1000, 5), rand(1000)))
println(EntropyInvariant.conditional_mutual_information(rand(5, 1000), rand(1, 1000), dim=2))
"""
