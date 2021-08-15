module Es

include("policy.jl")
include("util.jl")
include("noisetable.jl")
include("optim.jl")
include("vbn.jl")

using .Noise
using .Plcy
using .Util
using .Optimizer
using .Vbn

using IterTools
using LyceumMuJoCo
using MuJoCo 
using Flux
using Random


function run()
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

	p = Policy(nn, 0.02f0)
	obstat = Obstat(obssize, 1f-2)
	nt = NoiseTable(250000000, length(p.θ))
	opt = Adam(length(p.θ), 0.01f0)

	f = (nn; show_dist=false) -> eval_net(nn, env, mean(obstat), std(obstat); show_dist=show_dist)
	tot_steps = 0

	for i in 1:500
		@show "Gen $i"
		sm, sumsq, cnt = step(p, nt, f, 256, opt)
		if cnt != 0
			obstat = inc(obstat, sm, sumsq, cnt)
		end
		tot_steps += cnt

		@show tot_steps
		@show first(f(to_nn(p); show_dist=true))
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

	if show_dist @show LyceumMuJoCo._torso_x(env) end
	# @show rew step
	r, step, obs
end

function step(π, nt, f, n::Int, optim; l2coeff=0.005f0)  # TODO rename this because it mutates π
	results, obstats = evaluate(π, nt, f, n)
	ranked = rank(results)

	# TODO clean this up - minus positive noise fit from neg and adding up steps
	ranked = map((r) -> EsResult(first(r).fit - last(r).fit, first(r).ind, first(r).steps + last(r).steps), partition(ranked, 2))
	grad = l2coeff * π.θ - approxgrad(nt, ranked) ./ (n * 2)
	optimize!(π, optim, grad) # if this returns a new policy then Policy can be immutable
	
	@show Util.summary(results)

	obstats  # dunno if I like passing this out
end

function eval_one(pol::AbstractPolicy, nt, noise_ind, f)
	pπ, nπ, noise_ind = noiseify(pol, nt, noise_ind)  # todo make this

	pfit, psteps, pobs = f(to_nn(pπ))
	nfit, nsteps, nobs = f(to_nn(nπ))

	# These are vecs, how to not pass them back to master?
	sm, sumsq, cnt = zeros(size(first(pobs))), zeros(size(first(pobs))), 0
	if rand() < 0.01
		sm = sum(vcat(pobs, nobs))
		sumsq = sum(map(x -> x .^ 2, vcat(pobs, nobs)))
		cnt = length(pobs) + length(nobs)
		# all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, possibly less compute?
	end

	EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps), (sm, sumsq, cnt)
end

function evaluate(pol::AbstractPolicy, nt, f, n::Int)
	results = Vector{EsResult}()
	sm, sumsq, cnt = [], [], 0

	for i in 1:n
		pπ, nπ, noise_ind = noiseify(pol, nt)

		pfit, psteps, pobs = f(to_nn(pπ))
		nfit, nsteps, nobs = f(to_nn(nπ))

		if i == 1 sm, sumsq = zeros(size(first(pobs))), zeros(size(first(pobs))) end

		if rand() < 0.01
			sm .+= sum(vcat(pobs, nobs))
			sumsq .+= sum(map(x -> x .^ 2, vcat(pobs, nobs)))
			cnt += length(pobs) + length(nobs)
			# all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, possibly less compute?
		end
		push!(results, EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps))
	end

	results, (sm, sumsq, cnt)
end

function noiseify(pol::Policy, nt::NoiseTable)
	noise, ind = get_noise(nt)
	noise .*= pol.σ

	pos_θ = pol.θ .+ noise
	neg_θ = pol.θ .- noise

	Policy(pos_θ, pol.σ, pol._nn_maker), Policy(neg_θ, pol.σ, pol._nn_maker), ind
end

function noiseify(pol::Policy, nt::NoiseTable, ind::Int)
	noise, ind = get_noise(nt, ind)
	noise .*= pol.σ

	pos_θ = pol.θ .+ noise
	neg_θ = pol.θ .- noise

	Policy(pos_θ, pol.σ, pol._nn_maker), Policy(neg_θ, pol.σ, pol._nn_maker), ind
end

function approxgrad(nt::NoiseTable, rankedresults)
	fits = map(r-> r.fit, rankedresults)
	noises = map(r -> get_noise(nt, r.ind), rankedresults)
		
	sum([f .* n for (f, n) in zip(fits, noises)])
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

end
