abstract type AbstractPolicy end
function to_nn(::AbstractPolicy) end

mutable struct Policy{T} <: AbstractPolicy
    θ::AbstractVector{T}
    _re  # reconstructor
end

Policy(nn::Chain) = Policy{Float32}(Flux.destructure(nn)...)
to_nn(π::Policy) = π._re(π.θ)