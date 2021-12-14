abstract type AbstractPolicy end
function to_nn(::AbstractPolicy) end

mutable struct Policy{T} <: AbstractPolicy
    θ::AbstractVector{T}
    _re::Any  # reconstructor
end

Policy(nn::Chain) = Policy{Float32}(Flux.destructure(nn)...)

to_nn(π::Policy) = π._re(π.θ)

function bcast_policy!(::AbstractPolicy, ::ThreadComm) end # no need to do any sharing if not using mpi
bcast_policy!(π::Policy, comm::Comm) = π.θ[:] = bcast(π.θ, comm)

# not quite sure where to put this - don't want to make a whole file just for this
function forward(nn, x, obmean, obstd, rng)
    x = clamp.((x .- obmean) ./ obstd, -5, 5)
    out = nn(x)

    if rng !== nothing
        r = randn(rng, Float32, size(out)) .* 0.01
        out .+ r
    else
        out
    end
end