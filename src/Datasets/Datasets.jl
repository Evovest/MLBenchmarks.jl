module Datasets

using Random
using Arrow
using CSV
using DataFrames
import CategoricalArrays: categorical, levelcode, levels
using Statistics: mean, std
using StatsBase: median, tiedrank, quantile

using OpenML
using .Iterators: partition

export load_data, get_openml_data, data_recipe

include("utils.jl")
include("titanic.jl")
include("boston.jl")
include("year.jl")
include("microsoft.jl")
include("higgs_1M.jl")
include("sberbank.jl")
include("allstate_claims.jl")
include("creditcard.jl")

end
