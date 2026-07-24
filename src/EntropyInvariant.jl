module EntropyInvariant

# Import specific functions from dependencies
import Statistics: median, mean
import NearestNeighbors: KDTree, knn, inrangecount
import Distances: Chebyshev
import SpecialFunctions: gamma, digamma

# Export public API
export entropy, conditional_entropy, mutual_information, conditional_mutual_information,
       normalized_mutual_information, interaction_information, redundancy, unique, synergy,
       information_quality_ratio, mutual_information_ksg, conditional_mutual_information_ksg,
       redundancy_lattice, lattice_labels, moebius_atoms, coalition_mutual_information,
       isotonic_repair, mmi_redundancy, imin_redundancy, iccs_redundancy,
       specific_information, pid_lattice,
       RedundancyLattice

# Mathematical constant
const e = 2.718281828459045

# Include type definitions
include("types.jl")

# Include helper functions
include("helpers/utility_helpers.jl")
include("helpers/data_helpers.jl")
include("helpers/computation_helpers.jl")

# Include core functionality
include("ksg.jl")
include("entropy.jl")
include("mutual_information.jl")
include("advanced.jl")
include("pid.jl")
include("pid_lattice.jl")
include("optimized.jl")

end
