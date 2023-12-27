module Datasets

using Arrow
using DataFrames
using CSV
import CategoricalArrays: categorical, levelcode, levels
using Statistics: mean, std
using StatsBase: median, tiedrank

import ReadLIBSVM
import Random: seed!, randperm
import MLDatasets
import AWS: AWSCredentials, AWSConfig, @service
@service S3

export load_data

const aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
const aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

dataset_list = [:year, :higgs, :yahoo_ltrc, :msrank, :titanic, :boston]

abstract type Dataset{T} end

function load_data(x::Symbol; kwargs...)
    @assert x ∈ dataset_list
    data = load_data(Dataset{x})
    return data
end

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end


"""
    Uniformer

Project x over (min, max).
"""
struct Uniformer{S,V} <: Function
    edges::V
    nbins::Int
    min::S
    max::S
end

function Uniformer(x::AbstractVector; nbins=99, min=0.0, max=1.0, type="quantiles")
    if type == "quantiles"
        edges = sort(unique(quantile(skipmissing(x), (1:nbins-1) / nbins)))
    elseif type == "linear"
        edges = range(minimum(skipmissing(x)), maximum(skipmissing(x)), nbins + 1)[2:nbins]
    else
        @error "Invalid Binarizer type $type. Must be one of `quantile` or `linear`."
    end
    T = eltype(skipmissing(x))
    return Uniformer(edges, nbins, T(min), T(max))
end
function (m::Uniformer)(x::AbstractVector{T}) where {T}
    x_proj = zeros(T, length(x))
    @inbounds for i in eachindex(x)
        x_proj[i] =
            searchsortedfirst(m.edges, x[i]) / (m.nbins + 1) * (m.max - m.min) + m.min
    end
    return x_proj
end

function uniformer(
    df;
    vars_in,
    vars_out,
    nbins=99,
    min=0.0,
    max=1.0,
    type="quantiles",
)
    ops = [
        var_in => Uniformer(df[!, var_in]; nbins, min, max, type) => var_out for
        (var_in, var_out) in zip(vars_in, vars_out)
    ]
    return ops
end

include("year.jl")
include("higgs.jl")
include("titanic.jl")
include("boston.jl")

end
