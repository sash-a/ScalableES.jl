module ScalableES

using Distributed

numprocs = 4
procs = addprocs(numprocs, exeflags="--project")

@everywhere include("policy.jl")
@everywhere include("noisetable.jl")
@everywhere include("vbn.jl")
include("optim.jl")
include("util.jl")

@everywhere using .Plcy
@everywhere using .Noise
@everywhere using .Vbn
using .Optimizer
using .Util

@everywhere using ParallelDataTransfer
@everywhere using IterTools
@everywhere using LyceumMuJoCo
@everywhere using MuJoCo 
@everywhere using Flux
@everywhere using Random



# TODO test where @everywhere is needed
@everywhere include("core.jl")

# @everywhere using MuJoCo
# @everywhere using LyceumMuJoCo
# @everywhere using Flux
@everywhere using .Es

@everywhere mj_activate("/home/sasha/.mujoco/mjkey.txt")
@everywhere env = LyceumMuJoCo.HopperV2()
actsize = length(actionspace(env))
obssize = length(obsspace(env))

# nn = Chain(Dense(obssize, 256, tanh), 
# 			Dense(256, 256, tanh),
#  			Dense(256, actsize, tanh))

nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
            Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

Es.run(nn, env)

end