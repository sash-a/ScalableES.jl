abstract type AbstractResult{T} end

struct EsResult{T} <: AbstractResult{T}
    fit::T
    ind::Int
    steps::Int
end
make_result(fit::Float64, noise_ind::Int, steps::Int) = EsResult{Float64}(fit, noise_ind, steps)
make_result_vec(n::Int, ::Policy, ::AbstractComm) = Vector{EsResult{Float64}}(undef, n)
validresult(r::EsResult) = r.steps > 0


function rank(fs::AbstractVector{T}) where {T}
    len = length(fs)
    inds = sortperm(fs)
    ranked = zeros(Float32, len)
    ranked[inds] = (0:len-1)
    
    ranked ./ (len - 1) .- 0.5f0
end

function rank(results::AbstractVector{EsResult{T}}) where {T}
    """
    Ranks ES results using centered rank, then subtracts positive from negative fitnesses.  
    Assumes results vector is passed in as follows: [positive EsResult 1, negative EsResult 1, ...]
    """
    ranked = map((r, f) -> EsResult(f, r.ind, r.steps), results, rank((r -> r.fit).(results)))
    map(((p, n),) -> EsResult(p.fit - n.fit, p.ind, p.steps + n.steps), partition(ranked, 2))
end

sumsteps(res::AbstractVector{EsResult{T}}) where {T} = sum(map(r -> r.steps, res))

function StatsBase.summarystats(results::AbstractVector{EsResult{T}}) where {T}
    fits = map(r -> r.fit, results)
    StatsBase.summarystats(fits)
end

"""Reduces obstats in an all to all manner, but sends results to only the root proc"""
function share_results(
    local_results::AbstractVector{T},
    local_obstat::S,
    comm::Comm
) where {S<:AbstractObstat} where {T<:AbstractResult}
    results = gather(local_results, comm)
    obstat = allreduce(local_obstat, +, comm)

    results, obstat
end

function share_results(
    local_results::AbstractVector{T},
    local_obstat::S,
    ::ThreadComm,
) where {S<:AbstractObstat} where {T<:AbstractResult}
    local_results, local_obstat  # no need to do any sharing if not using mpi
end
