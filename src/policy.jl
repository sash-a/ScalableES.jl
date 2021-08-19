module Plcy

using Flux

export AbstractPolicy, Policy, to_nn

abstract type AbstractPolicy end
function to_nn(::AbstractPolicy) end

mutable struct Policy <: AbstractPolicy
    θ::Vector{Float32}
    _nn_maker
end

Policy(nn) = Policy(Flux.destructure(nn)...)

to_nn(π::Policy) = π._nn_maker(π.θ)

end