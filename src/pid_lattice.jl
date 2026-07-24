# N-source Partial Information Decomposition: the Williams & Beer redundancy lattice.
#
# `redundancy`/`unique`/`synergy` in pid.jl decompose I({X,Y}; Z) into exactly four
# terms and are hard-wired to two sources. This file provides the general N-source
# decomposition of Williams & Beer (2010), "Nonnegative Decomposition of Multivariate
# Information", in which I({X_1,...,X_N}; Z) splits into a lattice of partial-information
# atoms: 4 nodes for N=2, 18 for N=3, 166 for N=4.
#
# Functions included:
# - redundancy_lattice: the combinatorial structure (antichains + order + topology)
# - coalition_mutual_information: I(X_A; Z) for every coalition A of sources
# - isotonic_repair: project estimated coalition MIs back onto the monotone cone
# - specific_information: Williams & Beer's per-target-outcome information
# - pid_lattice: the atoms themselves, from continuous data or a discrete joint pmf
#
# Two redundancy measures are provided, because they answer differently:
# - :mmi   I_cap(a) = min over A in a of I(X_A; Z). Estimable from continuous data with
#          any of the package's estimators. Reduces EXACTLY to redundancy()/unique() at
#          N=2. Because it is a function of the 2^N-1 coalition MIs alone, its atoms are
#          a re-parameterization of those numbers, and its two-source unique atoms are
#          max(0, I_X - I_Y) and max(0, I_Y - I_X) -- so exactly one is nonzero by
#          construction. That is fine for asking how information is divided, and a poor
#          instrument for asking how much each source contributes; prefer
#          conditional_mutual_information for the latter.
# - :imin  Williams & Beer's original I_min, via specific information evaluated per
#          target outcome and averaged. Uses distributional structure the coalition MIs
#          discard and guarantees non-negative atoms (which :mmi does not for N >= 3),
#          but needs a discrete target, so it is supplied a joint pmf. Known to
#          over-credit redundancy (Harder, Salge & Polani 2013): on two-bit COPY it calls
#          two independent bits fully redundant. That is a property of the measure, and
#          the test suite asserts it is reproduced faithfully.
#
# NOTE ON VALIDATION. sum(atoms) == I_cap(top node) holds identically for ANY redundancy
# function, right or wrong -- it is a property of the Moebius inversion, not evidence the
# decomposition is correct. Correctness is established against published atom values on
# discrete toy distributions (AND, XOR, two-bit COPY, 3-way XOR), the Williams & Beer
# non-negativity theorem, and exact agreement with redundancy()/unique() at N=2. See
# test/runtests.jl.

"""
    RedundancyLattice

The Williams & Beer redundancy lattice for a fixed number of sources.

# Fields
- `n_sources::Int`: number of source variables.
- `nodes::Vector{Vector{UInt16}}`: lattice nodes. Each node is an antichain -- a
  canonical (sorted) collection of source coalitions, where a coalition is a bitmask over
  `1:n_sources` with bit `i` set iff source `i` belongs to it.
- `predecessors::Vector{Vector{Int}}`: `predecessors[i]` lists the indices of every
  STRICT predecessor of `nodes[i]` (the full down-set, not just immediate covers, which
  is what the Moebius inversion requires).
- `order::Vector{Int}`: a topological order of node indices in which every strict
  predecessor appears before the node itself.
"""
struct RedundancyLattice
    n_sources::Int
    nodes::Vector{Vector{UInt16}}
    predecessors::Vector{Vector{Int}}
    order::Vector{Int}
end

_coalitions(n::Int) = UInt16[UInt16(m) for m in 1:(2^n - 1)]
_issubset_mask(a::UInt16, b::UInt16) = (a & b) == a

function _is_antichain(c::AbstractVector{UInt16})
    for i in eachindex(c), j in eachindex(c)
        i == j && continue
        _issubset_mask(c[i], c[j]) && return false
    end
    return true
end

"""
    _antichains(n::Int) -> Vector{Vector{UInt16}}

Every non-empty antichain of the non-empty subsets of `1:n`. Enumerated by filtering all
`2^(2^n - 1)` collections, which is 128 candidates at n=3 and 32768 at n=4 -- immediate.
n >= 5 has 7579 nodes and would need a dedicated antichain generator; it is rejected
rather than attempted, since no estimator has the statistics to populate it.
"""
function _antichains(n::Int)
    1 <= n <= 4 || throw(ArgumentError(
        "redundancy_lattice supports 1 to 4 sources (n=5 has 7579 nodes and needs a " *
        "dedicated generator). Got n_sources=$n."))
    coals = _coalitions(n)
    out = Vector{Vector{UInt16}}()
    for sel in 1:(2^length(coals) - 1)
        c = UInt16[coals[i] for i in eachindex(coals) if (sel >> (i - 1)) & 1 == 1]
        _is_antichain(c) && push!(out, sort(c))
    end
    return out
end

"""
    _precedes(a, b) -> Bool

The Williams & Beer lattice order: `a <= b` iff every coalition in `b` contains some
coalition in `a`. Reflexive.
"""
_precedes(a::Vector{UInt16}, b::Vector{UInt16}) =
    all(B -> any(A -> _issubset_mask(A, B), a), b)

"""
    redundancy_lattice(n_sources::Int) -> RedundancyLattice

Build the Williams & Beer redundancy lattice for `n_sources` source variables:
4 nodes for 2 sources, 18 for 3, 166 for 4.

# Example
```julia
lat = redundancy_lattice(3)
length(lat.nodes)                      # 18
lattice_labels(lat, ["X", "Y", "Z"])   # {X}{Y}{Z}, {X}{Y}, ..., {XYZ}
```
"""
function redundancy_lattice(n_sources::Int)::RedundancyLattice
    nodes = _antichains(n_sources)
    n = length(nodes)
    preds = [Int[] for _ in 1:n]
    for i in 1:n, j in 1:n
        i == j && continue
        _precedes(nodes[j], nodes[i]) && push!(preds[i], j)
    end
    # In a partial order, b < a implies the down-set of b is a strict subset of that of
    # a, so ordering by down-set size is a valid topological order.
    order = sortperm(1:n, by = i -> (length(preds[i]), i))
    return RedundancyLattice(n_sources, nodes, preds, order)
end

"""
    lattice_labels(lat::RedundancyLattice, names::Vector{String}; sep::String = "") -> Vector{String}

Human-readable labels for every lattice node, e.g. `{X}{YZ}`. `sep` separates the members
inside a coalition, which is worth setting when source names are multi-character
(`sep="·"` gives `{X}{Y·Z}`).
"""
function lattice_labels(lat::RedundancyLattice, names::Vector{String}; sep::String = "")::Vector{String}
    length(names) == lat.n_sources || throw(ArgumentError(
        "expected $(lat.n_sources) names, got $(length(names))"))
    return map(lat.nodes) do a
        join(map(a) do A
            "{" * join([names[i] for i in 1:lat.n_sources if (A >> (i - 1)) & 1 == 1], sep) * "}"
        end, "")
    end
end
lattice_labels(lat::RedundancyLattice; kwargs...) =
    lattice_labels(lat, [string(i) for i in 1:lat.n_sources]; kwargs...)

"""
    moebius_atoms(lat::RedundancyLattice, I_cap::Function) -> Vector{Float64}

Partial-information atoms `Pi(a) = I_cap(a) - sum over strict predecessors b < a of Pi(b)`,
indexed like `lat.nodes`. `I_cap` maps a node (a `Vector{UInt16}` antichain) to its
cumulative redundancy.

`sum(atoms)` equals `I_cap` at the top node identically, for any `I_cap` -- that is how
Moebius inversion works and is not a check on correctness.
"""
function moebius_atoms(lat::RedundancyLattice, I_cap::Function)::Vector{Float64}
    atoms = fill(NaN, length(lat.nodes))
    for i in lat.order
        atoms[i] = I_cap(lat.nodes[i]) -
                   sum(Float64[atoms[j] for j in lat.predecessors[i]]; init = 0.0)
    end
    return atoms
end

"""
    coalition_mutual_information(sources, target; method, nbins, k, base, degenerate, dim) -> Dict{UInt16,Float64}

`I(X_A; Z)` for every non-empty coalition `A` of the source variables, keyed by the
coalition's bitmask.

`sources` is either a matrix whose rows (after `dim` canonicalisation) are the source
variables, or a vector of source vectors. `target` is a vector or single-row matrix.

For `method="inv_ksg"` the coalition block is normalised per dimension and passed to the
shared-radius KSG estimator, which is dimension-agnostic. Other methods go through the
joint-entropy chain rule; note that the `"inv"` plug-in entropy is only defined up to 3
total dimensions, so coalitions large enough to exceed that are rejected with a clear
error rather than a confusing one from deeper in the stack.
"""
function coalition_mutual_information(sources::Matrix{<:Real}, target::Matrix{<:Real};
                                      method::String = "inv_ksg", nbins::Int = 10,
                                      k::Int = 3, base::Real = e,
                                      degenerate::Bool = false, dim::Int = 1)::Dict{UInt16,Float64}
    src = ensure_columns_are_points(sources, dim)
    tgt = ensure_columns_are_points(target, dim)
    n_sources = size(src, 1)
    size(src, 2) == size(tgt, 2) || throw(ArgumentError(
        "sources and target must have the same number of points, got $(size(src,2)) and $(size(tgt,2))"))
    if method != "inv_ksg" && (n_sources + size(tgt, 1)) > 3
        throw(ArgumentError(
            "method=\"$method\" cannot evaluate a $(n_sources)-source coalition against a " *
            "$(size(tgt,1))-dimensional target: the plug-in entropy estimator is only " *
            "defined for 1-3 total dimensions. Use method=\"inv_ksg\", which is " *
            "dimension-agnostic."))
    end
    out = Dict{UInt16,Float64}()
    for A in _coalitions(n_sources)
        rows = [i for i in 1:n_sources if (A >> (i - 1)) & 1 == 1]
        block = src[rows, :]
        out[A] = if method == "inv_ksg"
            convert_to_base(_mi_ksg_from_normalized(_invariant_normalize_rows(block),
                                                    _invariant_normalize_rows(tgt), k), base)
        else
            entropy(block, method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) +
            entropy(tgt,   method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2) -
            entropy(vcat(block, tgt), method=method, nbins=nbins, k=k, base=base, degenerate=degenerate, dim=2)
        end
    end
    return out
end

# `dim` is accepted and ignored here: a vector of vectors has no row/column ambiguity to
# resolve. It must still be absorbed rather than forwarded, because callers such as
# pid_lattice pass `dim` through unconditionally, and a duplicate keyword would override
# the dim=2 layout this method constructs.
function coalition_mutual_information(sources::Vector{<:Vector{<:Real}},
                                      target::Vector{<:Real}; dim::Int = 2, kwargs...)
    mat = Matrix{Float64}(undef, length(sources), length(target))
    for (i, s) in enumerate(sources)
        length(s) == length(target) || throw(ArgumentError(
            "source $i has $(length(s)) points but target has $(length(target))"))
        mat[i, :] = s
    end
    return coalition_mutual_information(mat, reshape(Float64.(target), 1, :);
                                        dim = 2, kwargs...)
end

"""
    isotonic_repair(coalition_mi; mode = :isotonic, iterations = 500, tol = 1e-12) -> Dict{UInt16,Float64}

Project estimated coalition mutual informations onto the monotone cone
`I(X_A; Z) <= I(X_B; Z)` for `A` a subset of `B`.

True mutual information always satisfies this, but finite-sample kNN estimates measurably
do not: adding a source that is nearly redundant with those already present lowers the
estimate, because the neighbourhood is spread over an extra dimension. `:mmi` takes a
minimum over coalitions and the Moebius inversion takes differences between them, and both
assume the ordering holds -- unrepaired estimates therefore produce negative unique and
synergy atoms, which are mathematically impossible for true information.

- `:isotonic` -- least-squares projection onto the cone by Dykstra's algorithm over the
  pairwise constraints. Minimal distortion; the default.
- `:majorant` -- minimal monotone majorant, `I(B) <- max(I(B), max over A subset B of I(A))`.
  Simpler and only ever raises values, but distorts more.

This deliberately does NOT clamp to zero. Non-negativity is a constraint on the LEVEL of a
single estimate, whereas monotonicity is a constraint BETWEEN estimates -- a violated pair
is mutually inconsistent, so pooling loses nothing, but an estimate of -0.02 where the true
value is near zero is an ordinary fluctuation of an estimator that is deliberately
unbiased rather than truncated. Clamping it would bias every low-signal region upward and
destroy the symmetry of the noise. Returns a new dictionary; the input is not modified.
"""
function isotonic_repair(coalition_mi::AbstractDict{UInt16,<:Real};
                         mode::Symbol = :isotonic, iterations::Int = 500,
                         tol::Real = 1e-12)::Dict{UInt16,Float64}
    ks = sort(collect(keys(coalition_mi)), by = count_ones)
    v = Dict{UInt16,Float64}(A => Float64(coalition_mi[A]) for A in ks)

    if mode === :majorant
        for B in ks, A in ks
            A != B && _issubset_mask(A, B) && (v[B] = max(v[B], v[A]))
        end
        return v
    end
    mode === :isotonic || throw(ArgumentError("mode must be :isotonic or :majorant, got :$mode"))

    pairs = [(A, B) for A in ks for B in ks if A != B && _issubset_mask(A, B)]
    corr = Dict(p => 0.0 for p in pairs)
    for _ in 1:iterations
        maxshift = 0.0
        for p in pairs
            A, B = p
            a, b = v[A] - corr[p], v[B] + corr[p]
            if a > b                       # violated: pool to the midpoint
                mid = (a + b) / 2
                shift = a - mid
                v[A], v[B] = mid, mid
                corr[p] = shift
                maxshift = max(maxshift, abs(shift))
            else
                v[A], v[B] = a, b
                corr[p] = 0.0
            end
        end
        maxshift < tol && break
    end
    return v
end

"""
    mmi_redundancy(coalition_mi) -> Function

The minimum-mutual-information redundancy `I_cap(a) = min over A in a of I(X_A; Z)`, as a
function suitable for `moebius_atoms`. Monotone on the lattice provided `coalition_mi` is
monotone under subset inclusion -- run `isotonic_repair` on estimated values first.
"""
mmi_redundancy(coalition_mi::AbstractDict{UInt16,<:Real}) =
    (a::Vector{UInt16}) -> minimum(Float64[coalition_mi[A] for A in a])

"""
    specific_information(pmf::AbstractArray{<:Real}, coalition::Integer; base = 2) -> Vector{Float64}

Williams & Beer's specific information `I(Z=z; X_A) = sum_a p(a|z) log( p(z|a) / p(z) )`,
one value per target outcome.

`pmf` is the joint distribution over `(X_1, ..., X_N, Z)` with the target in the LAST
dimension. `coalition` is a bitmask over the sources.
"""
function specific_information(pmf::AbstractArray{<:Real}, coalition::Integer;
                              base::Real = 2)::Vector{Float64}
    p = Array{Float64}(pmf)
    n_sources = ndims(p) - 1
    A = UInt16(coalition)
    members = [i for i in 1:n_sources if (A >> (i - 1)) & 1 == 1]
    isempty(members) && throw(ArgumentError("coalition must be non-empty"))
    keep = vcat(members, n_sources + 1)
    drop = Tuple(setdiff(1:(n_sources + 1), keep))
    m = isempty(drop) ? p : dropdims(sum(p; dims = drop); dims = drop)
    nz = size(m)[end]
    pz = vec(sum(m; dims = Tuple(1:(ndims(m) - 1))))
    ma = reshape(m, :, nz)
    pa = vec(sum(ma; dims = 2))
    lg = x -> log(x) / log(base)
    out = zeros(Float64, nz)
    for z in 1:nz
        pz[z] <= 0 && continue
        s = 0.0
        for a in axes(ma, 1)
            paz = ma[a, z]
            (paz <= 0 || pa[a] <= 0) && continue
            s += (paz / pz[z]) * lg((paz / pa[a]) / pz[z])
        end
        out[z] = s
    end
    return out
end

"""
    imin_redundancy(pmf; base = 2) -> Function

Williams & Beer's original redundancy `I_min(a) = sum_z p(z) * min over A in a of I(Z=z; X_A)`,
as a function suitable for `moebius_atoms`.

Evaluated per target outcome and then averaged -- this is NOT a minimum over the
coalitions' total mutual informations (that is `mmi_redundancy`, a different measure that
happens to coincide at the two-source bottom node). Guarantees non-negative atoms, and
over-credits redundancy on two-bit COPY; both are properties of the measure.
"""
function imin_redundancy(pmf::AbstractArray{<:Real}; base::Real = 2)
    p = Array{Float64}(pmf)
    n_sources = ndims(p) - 1
    pz = vec(sum(p; dims = Tuple(1:n_sources)))
    cache = Dict{UInt16,Vector{Float64}}()
    spec(A) = get!(() -> specific_information(p, A; base = base), cache, A)
    return (a::Vector{UInt16}) -> begin
        s = [spec(A) for A in a]
        sum(pz[z] * minimum(t[z] for t in s) for z in eachindex(pz))
    end
end

"""
    _ccs_marginal_tables(p::Array{Float64}, masks) -> Dict

For each coalition bitmask in `masks`, the joint distribution over `(X_B, Z)` reshaped to
`(number of joint source outcomes, number of target outcomes)` together with the marginal
over `X_B`, plus a stride vector for turning a full realisation into the corresponding row.
Shared by every node so the marginals are built once per decomposition.
"""
function _ccs_marginal_tables(p::Array{Float64}, masks)
    n_sources = ndims(p) - 1
    dims = size(p)
    tables = Dict{UInt16,Tuple{Matrix{Float64},Vector{Float64},Vector{Int},Vector{Int}}}()
    for B in masks
        members = [i for i in 1:n_sources if (B >> (i - 1)) & 1 == 1]
        drop = Tuple(setdiff(1:n_sources, members))
        m = isempty(drop) ? p : dropdims(sum(p; dims = drop); dims = drop)
        nz = size(m)[end]
        joint = reshape(m, :, nz)
        marg = vec(sum(joint; dims = 2))
        # column-major strides over the retained source dimensions
        strides = Vector{Int}(undef, length(members))
        acc = 1
        for (j, i) in enumerate(members)
            strides[j] = acc
            acc *= dims[i]
        end
        tables[B] = (joint, marg, members, strides)
    end
    return tables
end

@inline function _ccs_row(idx::CartesianIndex, members::Vector{Int}, strides::Vector{Int})
    r = 1
    @inbounds for j in eachindex(members)
        r += (idx[members[j]] - 1) * strides[j]
    end
    return r
end

_ccs_sign(v::Float64, tol::Float64) = v > tol ? 1 : (v < -tol ? -1 : 0)

"""
    iccs_redundancy(pmf; base = 2, tol = 1e-12) -> Function

Ince's (2017) `I_ccs` redundancy -- "common change in surprisal" -- as a function suitable
for `moebius_atoms`. `pmf` is the joint distribution over `(X_1, ..., X_N, Z)` with the
target in the LAST dimension.

For a node `{A_1, ..., A_k}` it accumulates the pointwise co-information between the
coalition variables and the target,

    c = sum over non-empty T of the A's  of  (-1)^(|T|+1) * i(x_{union of T}; z)

where `i(x_B; z) = log p(x_B, z) / (p(x_B) p(z))` is the local mutual information, but
counts a realisation ONLY when every `i(x_{A_j}; z)` and `c` itself share the same sign.
That sign-agreement condition is the whole idea: it keeps only the surprisal change that
all coalitions genuinely hold in common.

Why it is worth having alongside `:mmi` and `:imin`:
- It fixes the two-bit COPY problem. For `Z = (X, Y)` with independent bits it gives
  `R = 0`, `U_X = U_Y = 1`, `Syn = 0`, the decomposition most measures agree is right,
  where `:imin` reports the two independent bits as fully redundant (`R = 1`).
- Its unique atoms are not winner-take-all. `:mmi` gives `max(0, I_X - I_Y)` and
  `max(0, I_Y - I_X)`, so exactly one is nonzero by construction; `I_ccs` lets both be
  positive simultaneously, which matters whenever the question is how much each source
  contributes rather than how the total divides.
- It is defined for any number of sources, unlike BROJA and the other
  optimisation-based measures, which do not extend cleanly past two.

In exchange, `I_ccs` is not guaranteed monotone on the lattice, so atoms can come out
negative. Ince argues these are meaningful rather than a defect; either way they are
expected behaviour here and are not repaired away.
"""
function iccs_redundancy(pmf::AbstractArray{<:Real}; base::Real = 2, tol::Real = 1e-12)
    p = Array{Float64}(pmf)
    n_sources = ndims(p) - 1
    pz = vec(sum(p; dims = Tuple(1:n_sources)))
    lg = x -> log(x) / log(base)
    ftol = Float64(tol)
    cache = Dict{Vector{UInt16},Float64}()

    return (a::Vector{UInt16}) -> get!(cache, a) do
        k = length(a)
        # every union of a non-empty sub-collection of the node's coalitions
        subsets = Vector{Tuple{UInt16,Int}}()
        for sel in 1:(2^k - 1)
            B = UInt16(0); cnt = 0
            for j in 1:k
                if (sel >> (j - 1)) & 1 == 1
                    B |= a[j]; cnt += 1
                end
            end
            push!(subsets, (B, cnt))
        end
        tables = _ccs_marginal_tables(p, Base.unique(vcat([s[1] for s in subsets], a)))

        total = 0.0
        for idx in CartesianIndices(p)
            w = p[idx]
            w <= 0 && continue
            z = idx[n_sources + 1]
            pz[z] <= 0 && continue

            local_mi = function (B::UInt16)
                joint, marg, members, strides = tables[B]
                r = _ccs_row(idx, members, strides)
                (joint[r, z] <= 0 || marg[r] <= 0) && return -Inf
                return lg(joint[r, z] / (marg[r] * pz[z]))
            end

            c = 0.0; ok = true
            for (B, cnt) in subsets
                v = local_mi(B)
                isfinite(v) || (ok = false; break)
                c += (isodd(cnt) ? 1.0 : -1.0) * v
            end
            ok || continue

            s0 = _ccs_sign(c, ftol)
            agree = true
            for A in a
                v = local_mi(A)
                if !isfinite(v) || _ccs_sign(v, ftol) != s0
                    agree = false; break
                end
            end
            agree && (total += w * c)
        end
        return total
    end
end

"""
    pid_lattice(sources, target; measure = :mmi, repair = :isotonic, method, nbins, k, base, degenerate, dim) -> Dict{String,Float64}
    pid_lattice(pmf::AbstractArray{<:Real}; measure = :imin, base = 2, names) -> Dict{String,Float64}

Full N-source partial information decomposition of `I({X_1,...,X_N}; Z)` over the
Williams & Beer redundancy lattice, returning every atom keyed by its label
(`"{1}{2}{3}"`, `"{12}"`, ...).

The first form estimates from continuous data with `measure = :mmi`. The second takes an
explicit discrete joint pmf, with the target in the last dimension, and uses
`measure = :imin` (Williams & Beer's `I_min`).

`repair` controls the monotonicity projection applied to the estimated coalition MIs
before decomposing (`:isotonic`, `:majorant`, or `:none`); see `isotonic_repair` for why
this is not optional in practice. Pass `names` to label atoms with variable names instead
of indices.

# Example
```julia
# continuous, three sources
atoms = pid_lattice([x1, x2, x3], z; names = ["a", "b", "c"])
atoms["{a}{b}{c}"]     # redundancy shared by all three individually
atoms["{abc}"]         # the top synergy atom

# discrete, from an explicit joint distribution -- two-input AND
pmf = zeros(2, 2, 2)
pmf[1,1,1] = pmf[1,2,1] = pmf[2,1,1] = 0.25; pmf[2,2,2] = 0.25
atoms = pid_lattice(pmf; names = ["X", "Y"])   # {X}{Y} = 0.3113, {XY} = 0.5 bits
```
"""
function pid_lattice(sources, target; measure::Symbol = :mmi, repair::Symbol = :isotonic,
                     names::Union{Nothing,Vector{String}} = nothing,
                     method::String = "inv_ksg", nbins::Int = 10, k::Int = 3,
                     base::Real = e, degenerate::Bool = false, dim::Int = 1)
    measure === :mmi || throw(ArgumentError(
        "measure=:$measure needs a discrete joint distribution; call " *
        "pid_lattice(pmf; measure=:imin) instead. Continuous data supports :mmi."))
    cmi = coalition_mutual_information(sources, target; method = method, nbins = nbins,
                                       k = k, base = base, degenerate = degenerate, dim = dim)
    n_sources = Int(log2(length(cmi) + 1))
    cmi = repair === :none ? cmi : isotonic_repair(cmi; mode = repair)
    lat = redundancy_lattice(n_sources)
    labels = names === nothing ? lattice_labels(lat) : lattice_labels(lat, names)
    return Dict(labels[i] => a for (i, a) in enumerate(moebius_atoms(lat, mmi_redundancy(cmi))))
end

function pid_lattice(pmf::AbstractArray{<:Real}; measure::Symbol = :imin, base::Real = 2,
                     names::Union{Nothing,Vector{String}} = nothing)
    measure in (:imin, :iccs) || throw(ArgumentError(
        "measure=:$measure is not defined on a discrete joint distribution; use :imin or :iccs."))
    ndims(pmf) >= 2 || throw(ArgumentError(
        "pmf needs at least 2 dimensions (>=1 source plus the target in the last dimension)"))
    lat = redundancy_lattice(ndims(pmf) - 1)
    labels = names === nothing ? lattice_labels(lat) : lattice_labels(lat, names)
    I_cap = measure === :imin ? imin_redundancy(pmf; base = base) :
                                iccs_redundancy(pmf; base = base)
    return Dict(labels[i] => a for (i, a) in enumerate(moebius_atoms(lat, I_cap)))
end
