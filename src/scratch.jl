using Distributed
p = addprocs(2, exeflags="--project")

# include("core.jl")

@everywhere begin
    using Pkg
    Pkg.activate(".")
    include("core.jl")
end

@everywhere begin
    using .Es
    using MuJoCo
    using LyceumMuJoCo
    using Flux
    using SharedArrays
end

mj_activate("/home/sasha/.mujoco/mjkey.txt")
@distributed for i in 1:nprocs()
    mj_activate("/home/sasha/.mujoco/mjkey.txt")
end
env = LyceumMuJoCo.HopperV2()
reset!(env)

actsize = length(actionspace(env))
obssize = length(obsspace(env))

# nn = Chain(Dense(obssize, 256, tanh), 
# 			Dense(256, 256, tanh),
#  			Dense(256, actsize, tanh))

nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
        Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
        Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

table_size = 25_000_000
σ = 0.02f0

noise = Es.Noise.make_noise(table_size, σ)
s_noise = SharedArray{Float32}((table_size,), init=s -> s[localindices(s)] = noise[localindices(s)], pids=procs())
nt = Es.Noise.NoiseTable(s_noise, length(first(Flux.destructure(nn))), σ)

# @show env
# @everywhere reset!(env)
@distributed for i in 1:nprocs()
    env = LyceumMuJoCo.HopperV2()
    reset!(env)
end

Es.run(nn, env, nt)