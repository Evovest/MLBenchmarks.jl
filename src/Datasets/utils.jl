abstract type Dataset{T} end

function load_data(x::Symbol; kwargs...)
    @assert x ∈ dataset_list
    data = load_data(Dataset{x}; kwargs...)
    return data
end

function read_arrow_aws(path; bucket="jeremiedb", aws_config=AWSConfig())
    raw = S3.get_object(bucket, path, Dict("response-content-type" => "application/octet-stream"); aws_config)
    df = DataFrame(Arrow.Table(raw))
    return df
end

function read_libsvm_aws(file::String; has_query=false, aws_config=AWSConfig())
    raw = S3.get_object("jeremiedb", file, Dict("response-content-type" => "application/octet-stream"); aws_config)
    return read_libsvm(raw; has_query)
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
    _nbins = length(edges) + 1
    T = eltype(skipmissing(x))
    return Uniformer(edges, _nbins, T(min), T(max))
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
