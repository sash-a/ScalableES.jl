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
using Flux
using Random

function run()
	env = LyceumMuJoCo.HopperV2()
	actsize = length(actionspace(env))
	obssize = length(obsspace(env))

	nn = Chain(Dense(obssize, 256, tanh), Dense(256, 256, tanh), Dense(256, actsize, tanh))
	p = Policy(nn, 0.02f0)
	obstat = Obstat(obssize, 1f-2)
	nt = NoiseTable(2500000, length(p.θ) - 1)
	opt = Adam(length(p.θ), 0.01f0)

	f = (nn) -> eval_net(nn, env, obstat)
	# f = (nn) -> eval_net
	for i in 1:1000
		@show "Gen $i"
		@time obs = step(p, nt, f, 500, opt)
		obstat = inc(obstat, sum(obs), sum(map(x -> x .^ 2, obs)), length(obs))
	end

end

function eval_net(nn::Chain, env, obstat)
	reset!(env)
	obs = []

	r = 0
	step = 0
	
	for i in 1:1000
		ob = getobs(env)
		act = forward(nn, ob, obstat)
		setaction!(env, act)
		step!(env)

		step += 1
		push!(obs, ob)
		r += getreward(env)
		if isdone(env) break end
	end

	# @show rew step
	r, step, obs
end

function step(π, nt, f, n::Int, optim; l2coeff=0.005f0)  # TODO rename this because it mutates π
	results, obs = evaluate(π, nt, f, n)

	ranked = rank(results)
	# TODO clean this up - minus positive noise fit from neg and adding up steps
	map((r) -> EsResult(first(r).fit - last(r).fit, first(r).ind, first(r).steps + last(r).steps), partition(ranked, 2))
	grad = l2coeff * π.θ - approxgrad(nt, ranked)
	optimize!(π, optim, grad) # if this returns a new policy then Policy can be immutable
	
	@show Util.summary(results)

	obs  # dunno if I like passing this out
end

function noiseify(pol::Policy, nt::NoiseTable)
	noise, ind = get_noise(nt)
	noise *= pol.σ

	pos_θ = pol.θ + noise
	neg_θ = pol.θ - noise
	Policy(pos_θ, pol.σ, pol._nn_maker), Policy(neg_θ, pol.σ, pol._nn_maker), ind
end

function evaluate(pol::AbstractPolicy, nt, f, n::Int)
	results = Vector{EsResult}()
	
	all_obs = []

	for i in 1:n
		pπ, nπ, noise_ind = noiseify(pol, nt)

		pfit, psteps, pobs = f(to_nn(pπ))
		nfit, nsteps, nobs = f(to_nn(nπ))

		if rand() < 0.01
			all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, could sum and sumsq each?
		end
		push!(results, EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps))
	end

	results, all_obs
end

function approxgrad(nt::NoiseTable, rankedresults)
	scalenoise(nt, rankedresults)
end

function scalenoise(nt::NoiseTable, rankedresults)
	fits = map(r-> r.fit, rankedresults)
	noises = map(r->get_noise(nt, r.ind), rankedresults)
		
	sum([f .* n for (f,n) in zip(fits, noises)])
end


function optimize!(π::Policy, optim, grad)
	π.θ = Optimizer.optimize(optim, grad)
end

function forward(nn, x, obstat; rng=Random.GLOBAL_RNG)
	# @show obstat.sum
	x = clamp.((x - mean(obstat)) ./ std(obstat), -5, 5)
	out = nn(x)
	
	r = zeros(size(out))
	if rng !== nothing
		r = randn(rng, Float32, size(out))
	end

	out + 0.01 * r
end

end
