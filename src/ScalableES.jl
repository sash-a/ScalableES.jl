include("core.jl")


using MuJoCo
using LyceumMuJoCo
using Flux
using .Es

mj_activate("/home/sasha/.mujoco/mjkey.txt")
env = LyceumMuJoCo.Walker2DV2()
actsize = length(actionspace(env))
obssize = length(obsspace(env))

# nn = Chain(Dense(obssize, 256, tanh), 
# 			Dense(256, 256, tanh),
#  			Dense(256, actsize, tanh))

nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
            Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

Es.run(nn, env)