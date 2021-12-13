using MPI

using Flux
using Random
using BenchmarkTools
using Future
using LinearAlgebra

using MuJoCo
using LyceumMuJoCo
using HrlMuJoCoEnvs

function stepmj(env, nn)
    ob = getobs(env)
    setaction!(env, nn(ob))
    step!(env)
end

function testmj(repeats, nns, envs, rngs)
    Threads.@threads for i in 1:repeats
        tid = Threads.threadid()
        rng = rngs[tid]
        env = envs[tid]
        nn = nns[tid]

        for _ in 1:100
            stepmj(env, nn)
        end
    end
end

# https://discourse.julialang.org/t/random-number-and-parallel-execution/57024
function parallel_rngs(rng::MersenneTwister, n::Integer)
    step = big(10)^20
    rngs = Vector{MersenneTwister}(undef, n)
    rngs[1] = copy(rng)
    for i = 2:n
        rngs[i] = Future.randjump(rngs[i-1], step)  # TODO step each by `procrank`
    end
    return rngs
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    nnodes = MPI.Comm_size(comm)

    LinearAlgebra.BLAS.set_num_threads(1)

    mj_activate("/home/sasha/.mujoco/mjkey.txt")

    mt = MersenneTwister()
    rngs = parallel_rngs(mt, Threads.nthreads())

    nns = ntuple(_ -> Chain(Dense(32, 256, tanh),
                            Dense(256, 256, tanh),
                            Dense(256, 8, tanh)), Threads.nthreads())

    envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntMazeEnv, Threads.nthreads())
    
    stepmj(first(envs), first(nns))  # warm up

    repeats = 720 ÷ nnodes
    @show repeats nnodes
    @btime testmj($repeats, $nns, $envs, $rngs) seconds = 10
end

main()