include("../src/ScalableES.jl")
using .ScalableES
using MuJoCo
using LyceumMuJoCo
using LyceumBase

using MPI
using Base.Threads

using Flux
using Dates
using Random
using ArgParse

using HrlMuJoCoEnvs


function threadedrun(runname, mjpath)
    savedfolder = joinpath(@__DIR__, "..", "saved", runname)
    if !isdir(savedfolder)
        mkdir(savedfolder)
    end

    mj_activate(mjpath)
    println("MuJoCo activated")

    println("n threads $(Threads.nthreads())")

    seed = 123  # auto generate and share this?
    envs = LyceumBase.tconstruct(HrlMuJoCoEnvs.Flagrun, "easier_ant.xml", Threads.nthreads(); interval=25, cropqpos=false, seed=seed)
    # envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntV2, Threads.nthreads())
    env = first(envs)
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))
    @show obssize

    nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
    println("nn created")
    t = now()
    run_es(runname, nn, envs, ScalableES.ThreadComm(); gens=600, episodes=5, steps=1000, npolicies=128)
    println("Total time: $(now() - t)")

end

s = ArgParseSettings()
@add_arg_table s begin
    "runname"
        required=true
        help="Name of the run for saving policies and tensorboard logs"
    "mjpath"
        required=true
        help="path/to/mujoco/mjkey.txt"
end
args = parse_args(s)
threadedrun(args["runname"], args["mjpath"])