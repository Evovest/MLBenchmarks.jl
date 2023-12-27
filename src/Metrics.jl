module Metrics

export mse, mae, logloss, accuracy, gini

using Statistics: mean, std

mse(p, y) = mean((p .- y) .^ 2)
mae(p, y) = mean(abs.(p .- y))

accuracy(p::Vector{Int}, y::Vector{Int}) = mean(p .== y)
accuracy(p::Vector{<:AbstractFloat}, y::Vector{Int}) = mean((p .> 0.5) .== y)
function accuracy(p::Matrix{AbstractFloat}, y::Vector{Int})
    _p = [findmax(row)[2] for row in eachrow(p)]
    return mean(_p .== y)
end

# function sigmoid(x::AbstractArray{T}) where {T<:AbstractFloat}
#     return sigmoid.(x)
# end
@inline function sigmoid(x::T) where {T<:AbstractFloat}
    @fastmath 1 / (1 + exp(-x))
end
logloss(p::AbstractVector, y::AbstractVector) = mean(-y .* log.(p) .+ (y .- 1) .* log.(1 .- p))

function gini_raw(p::AbstractVector, y::AbstractVector)
    _y = y .- minimum(y)
    if length(_y) < 2
        return 0.0
    end
    random = cumsum(ones(length(p)) ./ length(p)^2)
    y_sort = _y[sortperm(p)]
    y_cum = cumsum(y_sort) ./ sum(_y) ./ length(p)
    gini = sum(random .- y_cum)
    return gini
end

function gini(p::AbstractVector, y::AbstractVector)
    if length(y) < 2
        return 0.0
    end
    return gini_raw(y, p) / gini_raw(y, y)
end

end