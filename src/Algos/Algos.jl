using DataFrames
using StatsBase: sample

# Generic symbol entrypoint
function run_experiment(algo::Symbol, data, hyper_list; kwargs...)
    run_experiment(Val(algo), data, hyper_list; kwargs...)
end

function run_experiment(::Val{A}, data, hyper_list; kwargs...) where {A}
    throw(ArgumentError("No run_experiment implementation for algo=$(A)."))
end

include("evotrees.jl")
include("neurotrees.jl")
include("xgboost.jl")
include("lightgbm.jl")
include("catboost.jl")
