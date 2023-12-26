module Metrics

export mse, mae, logloss, accuracy

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

end