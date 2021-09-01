include("ScalableES.jl")
using .ScalableEs
using MuJoCo
using LyceumMuJoCo
using Flux
using MPI
using Dates
using Random

using HrlMuJoCoEnvs

function run()
    MPI.Init()
    println("MPI initialized")
    comm::MPI.Comm = MPI.COMM_WORLD
    node_comm::MPI.Comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, 0)

    if ScalableEs.isroot(node_comm)  # only activate mujoco once per node
        mj_activate("/home/sasha/.mujoco/mjkey.txt")
        println("MuJoCo activated")
    end

    seed = 123  # auto generate and share this?
    rng = MersenneTwister(seed)

    env::AbstractMuJoCoEnvironment = HrlMuJoCoEnvs.AntFlagrun(interval=200, rng=rng)
    actsize::Int = length(actionspace(env))
    obssize::Int = length(obsspace(env))

    nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
    
    println("nn created")
    t = now()
    run_es(nn, env, comm; episodes=5)
    println("Total time: $(now() - t)")
    MPI.Finalize()
end

run()