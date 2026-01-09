module EntropyInvariant

# Import specific functions from dependencies
import StatsBase: median, mean
import NearestNeighbors: KDTree, knn
import SpecialFunctions: gamma, digamma

# Export public API
export entropy, conditional_entropy, mutual_information, conditional_mutual_information,
       normalized_mutual_information, interaction_information, redundancy, unique, synergy,
       information_quality_ratio

# Mathematical constant
const e = 2.718281828459045

# Include type definitions
include("types.jl")

# Include helper functions
include("helpers/utility_helpers.jl")
include("helpers/data_helpers.jl")
include("helpers/computation_helpers.jl")

# Include core functionality
include("entropy.jl")
include("mutual_information.jl")
include("advanced.jl")
include("pid.jl")
include("optimized.jl")

end
