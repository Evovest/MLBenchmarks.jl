benchmarks = [
    "boston",
    "titanic",
    "year",
    "microsoft",
    "higgs_1M",
    "allstate_claims",
    "creditcard",
]

for benchmark in benchmarks
    @info "Launching $benchmark"
    run(`julia --project=. --threads=12 benchmarks/$(benchmark).jl`)
end
