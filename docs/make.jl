using Eeloo
using Documenter

DocMeta.setdocmeta!(Eeloo, :DocTestSetup, :(using Eeloo); recursive=true)

makedocs(;
    modules=[Eeloo],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Eeloo.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/Eeloo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Eeloo.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Eeloo.jl",
)
