module Es

# using Distributed

# numprocs = 4
# procs = addprocs(numprocs, exeflags="--project")
using Distributed


include("policy.jl")
include("noisetable.jl")
include("vbn.jl")
include("optim.jl")
include("util.jl")

using .Plcy
using .Noise
using .Vbn
using .Optimizer
using .Util

using ParallelDataTransfer
using IterTools
using LyceumMuJoCo
using MuJoCo 
using Flux
using Random

# TODO test where @everywhere is needed
function run(nn, env, nt::NoiseTable, gens=500, episodes=256, η=0.01f0)
	obssize = length(obsspace(env))

	pol = Policy(nn)  # the policy on the root process
	obstat = Obstat(obssize, 1f-2)
	# nt = NoiseTable(nt_size, length(pol.θ), σ)
	opt = Adam(length(pol.θ), η)

	f = (nn; show_dist=false) -> eval_net(nn, env, mean(obstat), std(obstat); show_dist=show_dist)
	# f = (model; show_dist=false) -> eval_net(model, env, mean(obstat), std(obstat); show_dist=show_dist)
	tot_steps = 0

	for i in 1:gens
		@show "Gen $i"
		@time sm, sumsq, cnt = step(pol, nt, f, episodes, opt, env, mean(obstat), std(obstat))
		if cnt != 0
			obstat = inc(obstat, sm, sumsq, cnt)
		end

		# sharing updated policy and obstat
		# sendto(procs, π=rootπ)
		# sendto(procs, obstat=root_obstat)

		tot_steps += cnt

		@show tot_steps
		@show first(f(to_nn(pol); show_dist=true))
		print("\n\n")
	end
end

function eval_net(nn::Chain, env, obmean, obstd; show_dist=false)
	reset!(env)
	obs = []

	r = 0
	step = 0

	for i in 1:500
		ob = getobs(env)
		act = forward(nn, ob, obmean, obstd)
		setaction!(env, act)
		step!(env)

		step += 1
		push!(obs, ob)  # propogate ob recording to here, don't have to alloc mem if not using obs
		r += getreward(env)
		if isdone(env) break end
	end

	# if show_dist @show LyceumMuJoCo._torso_x(env) end
	# @show rew step
	r, step, obs
end

function step(pol, nt, f, n::Int, optim, env, obmean, obstd; l2coeff=0.005f0)  # TODO rename this because it mutates π
	results, obstats = evaluate(pol, nt, f, n, env, obmean, obstd)
	ranked = rank(results)
	# TODO clean this up - minus positive noise fit from neg and adding up steps
	ranked = map((r) -> EsResult(first(r).fit - last(r).fit, first(r).ind, first(r).steps + last(r).steps), partition(ranked, 2))
	grad = l2coeff * pol.θ - approxgrad(nt, ranked) ./ (n * 2)
	optimize!(pol, optim, grad) # if this returns a new policy then Policy can be immutable
	
	@show Util.summary(results)

	obstats  # dunno if I like passing this out
end

function eval_one(pol::AbstractPolicy, noise::Vector, f, env, obmean, obstd)
	pπ, nπ, noise_ind = noiseify(pol, noise)

	pfit, psteps, pobs = f(to_nn(pπ))
	nfit, nsteps, nobs = f(to_nn(nπ))

	# These are vecs, how to not pass them back to master?
	sm, sumsq, cnt = nothing, nothing, 0
	if rand() < 0.01
		sm = sum(vcat(pobs, nobs))
		sumsq = sum(map(x -> x .^ 2, vcat(pobs, nobs)))
		cnt = length(pobs) + length(nobs)
		# all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, possibly less compute?
	end

	EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps), (sm, sumsq, cnt)
end

function evaluate(π::AbstractPolicy, nt::NoiseTable, f, n::Int, env, obmean, obstd)

	es_results = Vector{EsResult}()
	sm, sumsq, cnt = [], [], 0

	# sendto(procs, pol=π)
	sendto(Distributed.procs(), env=env)

	results = Distributed.pmap(1:n) do i
		eval_one(π, nt.noise, f, env, obmean, obstd)
	end

	for (pres, nres, (s, ssq, c)) in results
		if c != 0
			if cnt == 0 sm, sumsq = zeros(size(s)), zeros(size(s)) end  # initialize to zeros now that we know the size

			sm .+= s
			sumsq .+= ssq
			cnt += c
		end
		push!(es_results, pres, nres)
	end

	es_results, (sm, sumsq, cnt)
end

function noiseify(pol::Policy, noise::Vector)
	noise, ind = sample(noise, length(pol.θ))
	Policy(pol.θ .+ noise, pol._nn_maker), Policy(pol.θ .- noise, pol._nn_maker), ind
end

function approxgrad(nt::NoiseTable, rankedresults)
	fits = map(r-> r.fit, rankedresults)
	noises = map(r -> sample(nt, r.ind), rankedresults)
		
	sum([f .* n for (f, n) in zip(fits, noises)]) .* (1 / nt.σ)  # because noise already has std σ, which just messes with lr 
end

function optimize!(π::Policy, optim, grad)
	π.θ .+= Optimizer.optimize(optim, grad)
end

function forward(nn, x, obmean, obstd; rng=Random.GLOBAL_RNG)
	x = clamp.((x .- obmean) ./ obstd, -5, 5)
	out = nn(x)
	
	r = zeros(size(out))
	if rng !== nothing
		r = randn(rng, Float32, size(out)) .* 0.01
	end

	out .+ r
end

function realrun()
	global procs

	mj_activate("/home/sasha/.mujoco/mjkey.txt")
	env = LyceumMuJoCo.HopperV2()
	actsize = length(actionspace(env))
	obssize = length(obsspace(env))

	# nn = Chain(Dense(obssize, 256, tanh), 
	# 			Dense(256, 256, tanh),
	#  			Dense(256, actsize, tanh))

	nn = Chain(Dense(obssize, 256, tanh; initW=Flux.glorot_normal, initb=Flux.glorot_normal), 
            Dense(256, 256, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal),
            Dense(256, actsize, tanh;initW=Flux.glorot_normal, initb=Flux.glorot_normal))

	run(nn, env, procs)
end

end
