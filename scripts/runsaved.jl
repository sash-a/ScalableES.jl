include("../src/ScalableES.jl")
using .ScalableEs

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
    seed = rand(1:100000)
    env = HrlMuJoCoEnvs.AntFlagrun(interval=200, seed=seed)

    obmean, obstd = ScalableEs.mean(obstat), ScalableEs.std(obstat)
    states = collectstates(model, env, obmean, obstd)
    # modes = LyceumMuJoCoViz.EngineMode[LyceumMuJoCoViz.PassiveDynamics()]
    # push!(modes, LyceumMuJoCoViz.Trajectory([states]))

    # engine = LyceumMuJoCoViz.Engine(LyceumMuJoCoViz.default_windowsize(), env, Tuple(modes))
    # viewport = render(engine)
    
    visualize(env, controller = e -> act(e, model, obmean, obstd), trajectories=[states])
    states
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
    setaction!(e, ScalableEs.forward(model, obs, obmean, obstd))
end

function collectstates(nn::Chain, env, obmean, obstd)
    T = 1000
    states = Array(undef, statespace(env), T)

    reset!(env)

	r = 0

	for t in 1:T
		ob = getobs(env)
		act = ScalableEs.forward(nn, ob, obmean, obstd)
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

runsaved("offset-200-angletarg-targrew", "gen520")