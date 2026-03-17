using DataFrames
using StatsBase: sample
using Random: Xoshiro

# Generic symbol entrypoint
function run_experiment(algo::Symbol, data, hyper_list; kwargs...)
    run_experiment(Val(algo), data, hyper_list; kwargs...)
end

function run_experiment(::Val{A}, data, hyper_list; kwargs...) where {A}
    throw(ArgumentError("No run_experiment implementation for algo=$(A)."))
end

include("neurotabmodels/neurotabmodels.jl")
include("evotrees.jl")
include("xgboost.jl")
include("lightgbm.jl")
include("catboost.jl")
