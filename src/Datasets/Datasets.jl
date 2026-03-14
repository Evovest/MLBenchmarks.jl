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
include("yahoo.jl")
include("msrank.jl")
include("higgs.jl")

end
