abstract type AbstractPolicy end
function to_nn(::AbstractPolicy) end

mutable struct Policy{T} <: AbstractPolicy
    θ::AbstractVector{T}
    _re  # reconstructor
end

Policy(nn) = Policy{Float32}(Flux.destructure(nn)...)
Policy(nn, ::Comm) = Policy(nn)
function Policy(nn, ::ThreadComm)
    p, re = Flux.destructure(nn)
    T = typeof(first(p))
    shared_p = SharedVector{T}(length(p))
    shared_p[:] = p
    
    Policy{T}(shared_p, re)
end


to_nn(π::Policy) = π._re(π.θ)