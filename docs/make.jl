using Documenter, EntropyInvariant

makedocs(
    modules = [EntropyInvariant],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Félix Truong, Alexandre Giuliani",
    sitename = "EntropyInvariant.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/Entropy-Invariant/EntropyInvariant.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md"
    ],
)

deploydocs(
    repo = "https://github.com/Entropy-Invariant/EntropyInvariant.jl",
    devbranch = "main"
)