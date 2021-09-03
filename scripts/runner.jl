include("../src/ScalableES.jl")
using .ScalableEs
using MuJoCo
using LyceumMuJoCo
using LyceumBase

using MPI
using Base.Threads

using Flux
using Dates
using Random

using HrlMuJoCoEnvs

function run()
    MPI.Init()
    println("MPI initialized")
    comm::MPI.Comm = MPI.COMM_WORLD
    node_comm::MPI.Comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, 0)

    runname = "randpos-sqeuclidvel-vectarg-20ep"
    println("Run name: $(runname)")
    if ScalableEs.isroot(comm)
        savedfolder = joinpath(@__DIR__, "..", "saved", runname)
        if !isdir(savedfolder)
            mkdir(savedfolder)
        end
    end

    if ScalableEs.isroot(node_comm)  # only activate mujoco once per node
        mj_activate("/home/sasha/.mujoco/mjkey.txt")
        println("MuJoCo activated")
    
        println("n threads $(Threads.nthreads())")

        seed = 231  # auto generate and share this?
        envs = LyceumBase.tconstruct(HrlMuJoCoEnvs.Flagrun, "ant.xml", Threads.nthreads(); interval=50, seed=seed)
        # envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntV2, Threads.nthreads())
        env = first(envs)
        actsize::Int = length(actionspace(env))
        obssize::Int = length(obsspace(env))

        nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                    Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
        
        println("nn created")
        t = now()
        run_es(runname, nn, envs, comm; gens=1000, episodes=20, npolicies=128)
        println("Total time: $(now() - t)")
    end

    MPI.Finalize()
    println("Finalized!")
end

run()