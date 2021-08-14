module Plcy

using Flux

export AbstractPolicy, Policy, to_nn

abstract type AbstractPolicy end
function to_nn(::AbstractPolicy) end

mutable struct Policy <: AbstractPolicy
    θ::Vector{Float32}
    σ::Real
    _nn_maker
end

function Policy(nn, σ::Real) 
    θ, re = Flux.destructure(nn)
    Policy(θ, σ, re)
end

to_nn(π::Policy) = π._nn_maker(π.θ)

end