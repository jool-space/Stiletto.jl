using Eeloo
using Documenter

DocMeta.setdocmeta!(Eeloo, :DocTestSetup, :(using Eeloo); recursive=true)

makedocs(;
    modules=[Eeloo],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Eeloo.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Eeloo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Eeloo.jl",
    devbranch="main",
)
