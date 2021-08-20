module ScalableEs

include("core.jl")

using MuJoCo
using LyceumMuJoCo
using Flux
using MPI
using .Es

function run()
    MPI.Init()
    println("MPI INITED")
    comm = MPI.COMM_WORLD
    node_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, 0)

    mj_activate("/home/sasha/.mujoco/mjkey.txt")  # only run once on each node
    println("ACTIVATING MJ")
    env = LyceumMuJoCo.HopperV2()
    actsize = length(actionspace(env))
    obssize = length(obsspace(env))

    # nn = Chain(Dense(obssize, 256, tanh), 
    # 			Dense(256, 256, tanh),
    #  			Dense(256, actsize, tanh))

    nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))
    # nn = Dense(obssize, actsize, tanh)
    
    println("nn created")
    Es.run(nn, env, comm)

    MPI.Finalize()
end
end # module