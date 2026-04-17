
############################################################
# To be launched from MLBenchkark main folder:
# `julia benchmarks/launch.jl`
############################################################

BENCHMARKS = ["boston", "titanic", "year", "microsoft", "higgs_1M"]
nthreads = 12

for bench in BENCHMARKS
    @info "Running benchmark: $bench"
    script = joinpath(@__DIR__, "$bench.jl")
    cmd = `julia --project=. --threads=$nthreads $script`
    run(cmd)
end
