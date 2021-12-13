# it is expected that this file is called once per node so it should be run with the --map-by ppr:1:node arg to mpiexec:
# mpiexec --map-by ppr:1:node julia --project -t 24 scripts/mpirunner.jl runname, path/to/mjkey.txt
# for 1 mpiproc per node and 24 threads on each node

include("../src/ScalableES.jl")
using .ScalableES
using MuJoCo
using LyceumMuJoCo
using LyceumBase
using HrlMuJoCoEnvs

using MPI
using Base.Threads

using LinearAlgebra
using Flux
using Dates
using Random
using ArgParse

function mpirun(runname, mjpath)
    println("Run name: $(runname)")

    MPI.Init()
    comm::MPI.Comm = MPI.COMM_WORLD
    println("MPI initialized")

    if ScalableES.isroot(comm)
        savedfolder = joinpath(@__DIR__, "..", "saved", runname)
        if !isdir(savedfolder)
            mkdir(savedfolder)
        end
    end

    LinearAlgebra.BLAS.set_num_threads(1)

    mj_activate(mjpath)
    println("MuJoCo activated")

    println("n threads $(Threads.nthreads())")

    @show ScalableES.nprocs(comm) gethostname()

    seed = 123  # auto generate and share this?
    # envs = LyceumBase.tconstruct(HrlMuJoCoEnvs.Flagrun, "easier_ant.xml", Threads.nthreads(); interval=25, cropqpos=false, seed=seed)
    envs = LyceumMuJoCo.tconstruct(HrlMuJoCoEnvs.AntMazeEnv, Threads.nthreads())
    # envs = HrlMuJoCoEnvs.tconstruct(LyceumMuJoCo.HopperV2, Threads.nthreads())

    env = first(envs)
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))

    nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                x -> x .* 30)
    
    println("nn created")
    run_es(runname, nn, envs, comm; gens=30, episodes=5, steps=1000, npolicies=240)

    MPI.Finalize()
    println("Finalized!")
end

function main()
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
    mpirun(args["runname"], args["mjpath"])
end

main()