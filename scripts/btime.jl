using BenchmarkTools: @btime, @benchmark
using Random
using Future
using LinearAlgebra

using ThreadPools
using Base.Threads
using MPI: MPI, Comm

using Flux

using MuJoCo
using LyceumMuJoCo
using HrlMuJoCoEnvs

struct ThreadComm end  # used for when thread only mode

noderank(comm::Comm) = MPI.Comm_rank(comm)
noderank(::ThreadComm) = 0
procrank(comm) = (noderank(comm) * Threads.nthreads()) + Threads.threadid()
nnodes(comm::Comm) = MPI.Comm_size(comm)
nnodes(::ThreadComm) = 1

test_nn(nn, rng) = for i in 1:1000 nn(rand(rng, 5)) end
function test_mj(envs, rng)
    env = envs[Threads.threadid()]
    for i in 1:1000
        ob = getobs(env)
        setaction!(env, rand(rng, 8))
        step!(env)
    end
end

function setup(mj)
    envs = HrlMuJoCoEnvs.tconstruct(HrlMuJoCoEnvs.AntMazeEnv, Threads.nthreads())
    nn = Chain(Dense(5, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                Dense(256, 5, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
                x -> x * 30)
    mj ? (envs, test_mj) : (nn, test_nn)
end


function main()
    mj = true
    mpi = false

    comm = if mpi
        MPI.Init()
        MPI.COMM_WORLD
    else
        ThreadComm()
    end
    
    LinearAlgebra.BLAS.set_num_threads(1)

    arg, f = setup(mj)
    
    mt = MersenneTwister()
    rngs = parallel_rngs(mt, Threads.nthreads())

    ts = time_ns()
    n = Threads.nthreads() * 900 / nnodes(comm)

    @qthreads for _ in 1:n
        rng = rngs[Threads.threadid()]
        f(arg, rng)
    end
    # @show noderank(comm) MPI.identify_implementation() MPI.MPI_LIBRARY_VERSION_STRING
    println("[$(procrank(comm)), $(noderank(comm))] end: $((time_ns() - ts)/1e9)")
end

function parallel_rngs(rng::MersenneTwister, n::Integer)
    step = big(10)^20
    rngs = Vector{MersenneTwister}(undef, n)
    rngs[1] = copy(rng)
    for i = 2:n
        rngs[i] = Future.randjump(rngs[i-1], step)
    end
    return rngs
end

main()
