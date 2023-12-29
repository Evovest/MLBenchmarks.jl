module Metrics

export mse, mae, logloss, accuracy, gini, ndcg

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
    return gini_raw(p, y) / gini_raw(y, y)
end

function ndcg(p, y, k=10)
    k = min(k, length(p))
    p_order = partialsortperm(p, 1:k, rev=true)
    y_order = partialsortperm(y, 1:k, rev=true)
    _y = y[p_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    ndcg = sum(gains ./ discounts)

    y_order = partialsortperm(y, 1:k, rev=true)
    _y = y[y_order]
    gains = 2 .^ _y .- 1
    discounts = log2.((1:k) .+ 1)
    idcg = sum(gains ./ discounts)

    return idcg == 0 ? 1.0 : ndcg / idcg
end

end