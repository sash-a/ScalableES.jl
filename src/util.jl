function loginfo(tblogger, main_fit, results::AbstractVector{T}, tot_steps::Int, gen_time) where {T}
    ss = summarystats(results)
    println("Main fit: $main_fit")
    println("Total steps: $tot_steps")
    println("Time: $gen_time s")
    println(ss)
    with_logger(tblogger) do
        @info "" main_fitness = main_fit log_step_increment = 0
        @info "" summarystat = ss log_step_increment = 0
        @info "" gen_time_s = gen_time
        @info "" total_steps = tot_steps log_step_increment = 1
    end
end

function checkpoint(i::Int, name::String, p, obstat, opt, eval_fn, env, prev_eval, rng)
    if i % 10 == 0 || i == 1
        nn = to_nn(p)
        prev_eval = eval_fn(nn, env, rng, mean(obstat), std(obstat))
        println("Saving model with eval score $prev_eval")
        path = joinpath("saved", name, "policy-obstat-opt-gen$i.bson")
        @save path p obstat opt
    end

    prev_eval
end

function parallel_rngs(seed, n::Integer, comm)
    step = big(10)^20
    mt = MersenneTwister(seed)
    if noderank(comm) != 0  # first space out the mt on each node
        Future.randjump(mt, noderank(comm) * step)
    end
    parallel_rngs(mt, n, step) # then space them out on each thread
end

# https://discourse.julialang.org/t/random-number-and-parallel-execution/57024
function parallel_rngs(rng::MersenneTwister, n::Integer, step)
    rngs = Vector{MersenneTwister}(undef, n)
    rngs[1] = copy(rng)
    for i = 2:n
        rngs[i] = Future.randjump(rngs[i-1], step)
    end
    return rngs
end