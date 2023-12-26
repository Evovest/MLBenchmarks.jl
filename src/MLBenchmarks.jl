module MLBenchmarks

export load_data

include("Datasets/Datasets.jl")
using .Datasets

include("Metrics.jl")
using .Metrics

include("Algos/Algos.jl")

end # module MLBenchmarks
