# TODO this should be its own package

include("Path.jl")
include("NsEsResult.jl")
include("Novelty.jl")
using DataStructures

function run_nses(
    name::String,
    nns,
    envs,
    comm::Union{Comm,ThreadComm};
    gens = 150,
    npolicies = 256,
    steps = 500,
    episodes = 3,
    σ = 0.02f0,
    nt_size = 250000000,
    η = 0.01f0,
    behv_freq = 20,
    min_w = 0.8,
    seed = 123,
)
    @assert npolicies / nnodes(comm) % 2 == 0 "Num policies / num nodes must be even (policies:$npolicies, nodes:$(nnodes(comm)))"

    println("Running ScalableEs with novelty!")
    tblg = if ScalableES.isroot(comm)
        ScalableES.TBLogger("tensorboard_logs/$(name)", min_level = ScalableES.Logging.Info)
    else
        nothing
    end

    rngs = ScalableES.parallel_rngs(seed, ScalableES.nprocs(comm), comm)
    env = first(envs)
    obssize = length(obsspace(env))

    println("Creating policy")
    ps = [Policy(nn) for nn in nns]
    for p in ps
        bcast_policy!(p, comm)  # untested for mpi
    end

    println("Creating noise table")
    nt, win = NoiseTable(nt_size, length(first(ps).θ), σ, comm)

    obstat = Obstat(obssize, 1.0f-2)
    opt = isroot(comm) ? Adam(length(first(ps).θ), η) : nothing

    println("Initialization done")
    f = (nn, e, obmean, obstd) -> nsr_eval_net(nn, e, obmean, obstd, steps, episodes; record_behv_freq = behv_freq)
    behvfn =
        (nn, e, obmean, obstd) ->
            last(first(nsr_eval_net(nn, e, obmean, obstd, steps, episodes; record_behv_freq = behv_freq)))
    evalfn =
        (nn, e, obmean, obstd) -> first(
            first(
                nsr_eval_net(nn, e, obmean, obstd, steps, episodes; record_behv_freq = behv_freq, show_end_pos = true),
            ),
        )
    w_schedule = (w, fit, best_fit, tsb_fit) -> weight_schedule(w, fit, best_fit, tsb_fit; min_w = min_w)
    run_gens(
        gens,
        name,
        ps,
        nt,
        f,
        behvfn,
        evalfn,
        envs,
        npolicies,
        opt,
        obstat,
        tblg,
        steps,
        episodes,
        w_schedule,
        behv_freq,
        rngs,
        comm,
    )

    for (i, p) in enumerate(ps)
        @save joinpath("saved", name, "policy-obstat-opt-final-$i.bson") p obstat opt
    end

    if win !== nothing
        MPI.free(win)
    end
end

"""
## Arguments

- `policies`:   all policies in the population
- `npolicies`:  the number of policies to rollout in a generation
- `w_schedule`: a function that changes the weighting of fitness/novelty depending on policy performance
"""
function run_gens(
    n::Int,
    name::String,
    policies::Vector{T},
    nt::NoiseTable,
    fn,
    behv_fn,
    eval_fn,
    envs,
    npolicies::Int,
    opt::AbstractOptim,
    obstat::AbstractObstat,
    logger,
    steps::Int,
    episodes::Int,
    w_schedule,
    record_behv_freq,
    rngs,
    comm::Union{Comm,ThreadComm},
) where {T<:AbstractPolicy}
    tot_steps = 0
    eval_score = -Inf
    env = first(envs)

    archive = init_archive(policies, (nn) -> behv_fn(nn, env, first(rngs), mean(obstat), std(obstat)))
    novs = [a.novelty for a in archive]  # archive has 1 nov for each policy
    tsb_fit = 0  # time since best fitness
    best_fit = -Inf
    w = 1.0

    for i = 1:n
        t = now()
        # selecting the current policy. Higher novelty=higher selection chance
        p_idx = sample(1:length(policies), Weights(novs))
        p = policies[p_idx]
        # I dislike this and would rather f is simply passed to this function, 
        #  but that doesn't allow for obmean and obstd to be updated in the 
        #  partial function f
        f = (nn, e, rng) -> fn(nn, e, rng, mean(obstat), std(obstat))
        res, gen_obstat =
            step_es(p, nt, f, envs, npolicies, opt, archive, episodes, steps, record_behv_freq, w, rngs, comm)
        obstat += gen_obstat

        update_archive!(archive, p, 10, (nn) -> behv_fn(nn, env, first(rngs), mean(obstat), std(obstat)))
        novs[p_idx] = archive[end].novelty  # updates chance of selecting policy again


        # calculating stats once
        perf = performance(res)
        w, best_fit, tsb_fit = w_schedule(w, perf, best_fit, tsb_fit)

        if isroot(comm)
            println("\nGen $i")

            gen_eval = checkpoint(i, name, p, obstat, opt, eval_fn, env, eval_score, first(rngs))
            eval_score = max(eval_score, gen_eval)

            tot_steps += sumsteps(res)
            loginfo(logger, gen_eval, res, tot_steps, t, w, tsb_fit)
        end
    end
end

function step_es(
    π::AbstractPolicy,
    nt,
    f,
    envs,
    n::Int,
    optim,
    archive,
    rollouts,
    steps,
    interval,
    w::Float64,
    rngs,
    comm::Union{Comm,ThreadComm};
    l2coeff = 0.005f0,
)  # TODO rename this because it mutates π
    results = make_result_vec(n, π, rollouts, steps, interval, comm)
    obstat = make_obstat(length(obsspace(first(envs))), π)

    local_results, obstat = evaluate(π, nt, f, envs, n ÷ nnodes(comm) ÷ 2, results, obstat, rngs, comm)
    results, obstat = share_results(local_results, obstat, comm)

    if isroot(comm)
        @show typeof(archive)
        results = novelty(results, archive, 10)
        ranked = rank(results, w)
        optimize!(π, ranked, nt, optim, l2coeff)  # if this returns a new policy then Policy can be immutable
    end

    bcast_policy!(π, comm)

    results, obstat
end

function nsr_eval_net(
    nn::Chain,
    env,
    obmean,
    obstd,
    steps::Int,
    episodes::Int;
    record_behv_freq = 20,
    show_end_pos = false,
)
    obs = Vector{Vector{Float64}}()
    paths = Vector{Path}()
    r = 0.0
    step = 0

    for e = 1:episodes
        path = Path()
        LyceumMuJoCo.reset!(env)
        for i = 0:steps-1
            ob = getobs(env)
            act = forward(nn, ob, obmean, obstd)
            setaction!(env, act)
            step!(env)

            step += 1
            push!(obs, ob)  # propogate ob recording to here, don't have to alloc mem if not using obs

            if record_behv_freq > 0 && i % record_behv_freq == 0
                push!(path, Vec2(HrlMuJoCoEnvs._torso_xy(env)...))
            end

            if record_behv_freq < 0 && i == steps - 1  # if behv_freq == -1 then only record final position
                push!(path, Vec2(HrlMuJoCoEnvs._torso_xy(env)...))
            end

            r += getreward(env)
            if isdone(env)
                break
            end
        end

        if record_behv_freq > 0
            # make all paths same length by repeating last element
            path = vcat(path, fill(path[end], (steps ÷ record_behv_freq) - length(path)))
        end
        push!(paths, path)
    end
    # @show rew step
    if show_end_pos
        pos = HrlMuJoCoEnvs._torso_xy(env)
        @show pos euclidean(pos, [0, 0])
        print("\n\n")
    end
    (r / episodes, paths), step, obs
end
