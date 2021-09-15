include("../src/ScalableES.jl")
using .ScalableES

using MuJoCo
using LyceumMuJoCo
using LyceumMuJoCoViz
using HrlMuJoCoEnvs
using Distances

using Flux

using BSON:@load
using StaticArrays

function runsaved(runname, suffix)
    @load "saved/$(runname)/model-obstat-opt-$suffix.bson" model obstat opt
    
    mj_activate("/home/sasha/.mujoco/mjkey.txt")
    env = first(LyceumBase.tconstruct(HrlMuJoCoEnvs.Flagrun, "ant.xml", 1; interval=200, seed=nothing))

    # nob = ScalableES.Obstat(length(obstat.sum), 1f-2)
    # obmean, obstd = ScalableES.mean(nob), ScalableES.std(nob)
    obmean, obstd = ScalableES.mean(obstat), ScalableES.std(obstat)
    states = collectstates(model, env, obmean, obstd)
    # modes = LyceumMuJoCoViz.EngineMode[LyceumMuJoCoViz.PassiveDynamics()]
    # push!(modes, LyceumMuJoCoViz.Trajectory([states]))

    # engine = LyceumMuJoCoViz.Engine(LyceumMuJoCoViz.default_windowsize(), env, Tuple(modes))
    # viewport = render(engine)
    test_rew,_,_ = ScalableES.eval_net(model, env, obmean, obstd, 500, 1000)
    @show test_rew
    visualize(env, controller = e -> act(e, model, obmean, obstd), trajectories=[states])
    # states
end

function render(e::LyceumMuJoCoViz.Engine)
    w, h = GLFW.GetFramebufferSize(e.mngr.state.window)
    rect = mjrRect(Cint(0), Cint(0), Cint(w), Cint(h))
    mjr_render(rect, e.ui.scn, e.ui.con)
    e.ui.showinfo && overlay_info(rect, e)
    GLFW.SwapBuffers(e.mngr.state.window)
    return rect
end

function act(e, model, obmean, obstd)
    obs = getobs(e)
    if sqeuclidean(e.target, HrlMuJoCoEnvs._torso_xy(e)) <= 1
        println("REACHED TARGET")
    end
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

        if sqeuclidean(env.target, HrlMuJoCoEnvs._torso_xy(env)) < 1
            println("Got target")
        end

        r += getreward(env)
		if isdone(env)
            println("dead")
            break
        end
	end

	@show r geteval(env)
	states
end

runsaved("flagrun-one_minus_dist_percent", "gen600")