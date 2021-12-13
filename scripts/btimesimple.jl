using Flux
using Random
using Base.Threads
using BenchmarkTools
using Future

using SharedArrays

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
    for i in 1:repeats
        tid = threadid()
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
    LinearAlgebra.BLAS.set_num_threads(1)

    mt = MersenneTwister()
    rngs = parallel_rngs(mt, Threads.nthreads())

    nns = ntuple(_ -> Chain(Dense(32, 256, tanh),
                            Dense(256, 256, tanh),
                            Dense(256, 8, tanh)), Threads.nthreads())

    envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntMazeEnv, Threads.nthreads())
    
    stepmj(first(envs), first(nns))  # warm up

    repeats = 256
    @btime testmj($repeats, $nns, $envs, $rngs)
end

main()