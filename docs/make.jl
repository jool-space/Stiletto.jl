using Stiletto
using Documenter

DocMeta.setdocmeta!(Stiletto, :DocTestSetup, :(using Stiletto); recursive=true)

makedocs(;
    modules=[Stiletto],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Stiletto.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/Stiletto.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Stiletto.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Stiletto.jl",
)
