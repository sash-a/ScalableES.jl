include("../src/ScalableES.jl")
using .ScalableES

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
using HrlMuJoCoEnvs
using Distances

using Flux

using BSON: @load, load
using StaticArrays
using SharedArrays
using Distributed

function runsaved(runname, suffix)
    @load "saved/$(runname)/policy-obstat-opt-$suffix.bson" p obstat opt
    
    nn = Chain(Dense(32, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 8, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                x -> x .* 30)

    @show size(p.θ)
    @show sum(length, params(nn))
    
    mj_activate("/home/sasha/.mujoco/mjkey.txt")
    env = HrlMuJoCoEnvs.AntFlagrun(;cropqpos=false)
    @show obsspace(env) length(obsspace(env)) getsim(env).m.nq getsim(env).m.nv
    @show size(getobs(env))

    pol = ScalableES.Policy(nn)
    pol.θ = p.θ

    model = ScalableES.to_nn(pol)
    @show size(first(model.layers).W)

    obmean, obstd = ScalableES.mean(obstat), ScalableES.std(obstat)
    @show size(obmean)

    states = collectstates(model, env, obmean, obstd)
    test_rew,_,_ = ScalableES.eval_net(model, env, obmean, obstd, 500, 1000)
    @show test_rew
    visualize(env, controller = e -> act(e, model, obmean, obstd), trajectories=[states])
    # states
end

function act(e, model, obmean, obstd)
    obs = getobs(e)
    # if sqeuclidean(e.target, HrlMuJoCoEnvs._torso_xy(e)) <= 1
    #     println("REACHED TARGET")
    # end
    setaction!(e, ScalableES.forward(model, obs, obmean, obstd))
end

function collectstates(nn::Chain, env, obmean, obstd)
    T = 1000
    states = Array(undef, statespace(env), T)

    reset!(env)

	r = 0

	for t in 1:T
		ob = getobs(env)
		act = ScalableES.forward(nn, ob, obmean, obstd)
		setaction!(env, act)
		step!(env)

        states[:, t] .= getstate(env)

        # if sqeuclidean(env.target, HrlMuJoCoEnvs._torso_xy(env)) < 1
        #     println("Got target")
        # end

        r += getreward(env)
		if isdone(env)
            println("dead")
            break
        end
	end

	@show r geteval(env)
	states
end

runsaved("remote/flagrun-i25-pretrain-fix-x30/", "gen600")