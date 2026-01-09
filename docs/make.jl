using Documenter, EntropyInvariant

makedocs(
    modules = [EntropyInvariant],
    authors = "Félix Truong, Alexandre Giuliani",
    sitename = "EntropyInvariant.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://github.com/Entropy-Invariant/EntropyInvariant.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorial" => "tutorial.md",
        "Theory" => "theory.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/Entropy-Invariant/EntropyInvariant.jl",
    devbranch = "main",
)
