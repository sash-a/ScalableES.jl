module ScalableES

using MPI: MPI, Comm
using Base.Threads
using ThreadPools
using SharedArrays

using LyceumMuJoCo
using MuJoCo
using HrlMuJoCoEnvs

using Flux
using StaticArrays
using Distributions
using StatsBase
import Statistics: mean, std

using Dates
using BSON: @save
using TensorBoardLogger, Logging
using IterTools
using Distances
using Random
using Future

include("mpi.jl")
include("policy.jl")
include("obstat.jl")
include("noise_table.jl")
include("result.jl")
include("optimizer.jl")
include("util.jl")
include("env.jl")

export run_es, run_nses

"""
Runs evolution strategy

# Arguments

- `name::String`: name of the run (for logging)
- `nn`: the Flux neural network that will be transformed into a `Policy`
- `envs`: the environment to evaluate, one per thread
- `comm::AbstractComm`: communicator, either `ThreadComm` or an MPI `Comm`
- `gens::Int`: number of generations to run ES
- `npolicies::Int`: number of policies to evaluate per generation
- `steps::Int`: the step horizon for each episodes
- `episodes::Int`: number of episodes to run each policy
- `σ::Float32`: the standard deviation of the noise applied to the policy params
- `nt_size::Int`: number of elements in the noise table
- `η::Float32`: learning rate
- `seed`: seed for the noise table and rngs
"""
function run_es(
    name::String,
    nn,
    envs,
    comm::AbstractComm;
    gens::Int = 150,
    npolicies::Int = 256,
    steps::Int = 500,
    episodes::Int = 3,
    σ::Float32 = 0.02f0,
    nt_size::Int = 250000000,
    η::Float32 = 0.01f0,
    seed = 123
)
    @assert npolicies / nnodes(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(nnodes(comm)))"

    println("Running ScalableEs")
    tblg = isroot(comm) ? TBLogger("tensorboard_logs/$(name)", min_level = Logging.Info) : nothing

    env = first(envs)
    obssize = length(obsspace(env))

    println("Creating policy")
    p = ScalableES.Policy(nn)
    bcast_policy!(p, comm)

    println("Creating rngs")
    rngs = parallel_rngs(seed, nprocs(comm), comm)
    
    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(p.θ), σ, comm; seed=seed)

    obstat = Obstat(obssize, 1.0f-2)
    opt = isroot(comm) ? Adam(length(p.θ), η) : nothing

    println("Initialization done")
    f = (nn, e, rng, obmean, obstd) -> eval_net(nn, e, obmean, obstd, steps, episodes, rng)
    evalfn = (nn, e, rng, obmean, obstd) -> first(eval_net(nn, e, obmean, obstd, steps, episodes, rng))

    f(to_nn(p), first(envs), first(rngs), mean(obstat), std(obstat))

    t = time_ns()
    run_gens(gens, name, p, nt, f, evalfn, envs, npolicies, opt, obstat, tblg, rngs, comm)
    println("Total time: $((time_ns() - t) / 1e9) s")

    @save joinpath("saved", name, "policy-obstat-opt-final.bson") p obstat opt

    if win !== nothing
        MPI.free(win)
    end
end

function run_gens(
    n::Int,
    name::String,
    p::AbstractPolicy,
    nt::NoiseTable,
    fn,
    eval_fn,
    envs,
    npolicies::Int,
    opt::Union{AbstractOptim,Nothing},
    obstat::AbstractObstat,
    logger,
    rngs,
    comm::Union{Comm,ThreadComm},
)
    tot_steps = 0
    eval_score = -Inf
    env = first(envs)

    for i = 1:n
        t = time_ns()
        # I dislike this and would rather f is simply passed to this function, 
        #  but that doesn't allow for obmean and obstd to be updated in the 
        #  partial function f
        f = (nn, e, rng) -> fn(nn, e, rng, mean(obstat), std(obstat))
        res, gen_obstat = step_es(p, nt, f, envs, npolicies, opt, rngs, comm)
        obstat += gen_obstat

        gt = (time_ns() - t) / 1e9

        if isroot(comm)
            println("\n\nGen $i")

            gen_eval = checkpoint(i, name, p, obstat, opt, eval_fn, env, eval_score, first(rngs))
            eval_score = max(eval_score, gen_eval)

            tot_steps += sumsteps(res)
            loginfo(logger, gen_eval, res, tot_steps, gt)
        end
    end
end

function step_es(π::AbstractPolicy, nt, f, envs, n::Int, optim, rngs, comm::AbstractComm; l2coeff = 0.005f0)  # TODO rename this because it mutates π
    local_results, obstat = evaluate(π, nt, f, envs, n ÷ nnodes(comm), rngs, comm)
    results, obstat = share_results(local_results, obstat, comm)

    if isroot(comm)
        ranked = rank(results)
        optimize!(π, ranked, nt, optim, l2coeff)  # if this returns a new policy then Policy can be immutable
    end

    bcast_policy!(π, comm)
    results, obstat
end

"""Results and obstat are empty containers of the correct type"""
function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int, results, obstat, rngs, comm)
    l = ReentrantLock()

    @qthreads for i = 1:n
        env = envs[Threads.threadid()]
        rng = rngs[Threads.threadid()]

        pπ, nπ, noise_ind = noiseify(pol, nt, rng)

        pnn = to_nn(pπ)
        nnn = to_nn(nπ)

        pfit, psteps, pobs = f(pnn, env, rng)
        nfit, nsteps, nobs = f(nnn, env, rng)

        if rand(rng) < 0.001
            Base.@lock l begin
                obstat = add_obs(obstat, pobs)
                obstat = add_obs(obstat, nobs)
            end
        end

        @inbounds results[i*2-1] = make_result(pfit, noise_ind, psteps)
        @inbounds results[i*2] = make_result(nfit, noise_ind, nsteps)

    end

    results, obstat
end

function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int, rngs, comm::AbstractComm)
    # TODO store fits as Float32
    results = make_result_vec(n, pol, comm)  # [positive EsResult 1, negative EsResult 1, ...]
    obstat = make_obstat(length(obsspace(first(envs))), pol)

    evaluate(pol, nt, f, envs, n ÷ 2, results, obstat, rngs, comm)  # ÷ 2 because sampling pos and neg
end

function approxgrad(π::AbstractPolicy, nt::NoiseTable, rankedresults::Vector{EsResult{T}}) where {T<:AbstractFloat}
    fits = map(r -> r.fit, rankedresults)
    noises = map(r -> sample(nt, r.ind, length(π.θ)), rankedresults)

    sum([f .* n for (f, n) in zip(fits, noises)]) .* (1 / nt.σ)  # noise already has std σ, which just messes with lr 
end

include("novelty/ScalableNsEs.jl")

end  # module
