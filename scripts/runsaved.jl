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
using Random

function runsaved(pol, obstat, opt)
    mj_activate("/home/sasha/.mujoco/mjkey.txt")
    env = HrlMuJoCoEnvs.AntFlagrun(;cropqpos=false)
    model = ScalableES.to_nn(pol)
    obmean, obstd = ScalableES.mean(obstat), ScalableES.std(obstat)
    visualize(env, controller = e -> act(e, model, obmean, obstd))
end

function act(e, model, obmean, obstd)
    obs = getobs(e)
    # if sqeuclidean(e.target, HrlMuJoCoEnvs._torso_xy(e)) <= 1
    #     println("REACHED TARGET")
    # end
    setaction!(e, ScalableES.forward(model, obs, obmean, obstd, Random.GLOBAL_RNG))
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

runname = "flagrun-pretrained"
gen = 950
# /home/sasha/Documents/es/ScalableES/saved/flagrun-pretrained/policy-obstat-opt-gen1.bson
@load "saved/$(runname)/policy-obstat-opt-gen$gen.bson" p obstat opt
runsaved(p, obstat, opt)