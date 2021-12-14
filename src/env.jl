function eval_net(nn::Chain, env, obmean, obstd, steps::Int, episodes::Int, rng)
    obs = Vector{Vector{Float64}}(undef, steps * episodes)
    r = 0.0
    step = 0

    for e = 1:episodes
        LyceumMuJoCo.reset!(env)
        for i = 1:steps
            ob = getobs(env)
            act = forward(nn, ob, obmean, obstd, rng)
            setaction!(env, act)
            step!(env)

            step += 1
            obs[step] = ob
            r += getreward(env)
            if isdone(env)
                break
            end
        end
    end

    r / episodes, step, obs[1:step]
end
