module ScalableEs

include("core.jl")

using MuJoCo
using LyceumMuJoCo
using Flux
using MPI
using .Es

function run()
    MPI.Init()
    println("MPI initialized")
    comm = MPI.COMM_WORLD
    node_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, 0)

    if Es.isroot(node_comm)  # only activate mujoco once per node
        mj_activate("/home/sasha/.mujoco/mjkey.txt")
        println("MuJoCo activated")
    end

    env = LyceumMuJoCo.HopperV2()
    actsize = length(actionspace(env))
    obssize = length(obsspace(env))

    nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
    
    println("nn created")
    Es.run(nn, env, comm)

    MPI.Finalize()
end
end # module