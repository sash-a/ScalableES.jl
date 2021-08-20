module Es

include("policy.jl")
include("noisetable.jl")
include("optim.jl")
include("vbn.jl")
include("mpiutils.jl")
include("util.jl")

using .Plcy
using .Noise
using .Optimizer
using .Vbn
using .MpiUtils
using .Util

using MPI: MPI, Comm
using IterTools
using LyceumMuJoCo
using MuJoCo 
using Flux
using Random
using StaticArrays
using Dates

function run(nn, env, comm::Comm, gens=500, episodes=256, σ=0.02f0, nt_size=250000000, η=0.01f0)
	println("RUNNING")

	actsize = length(actionspace(env))
	obssize = length(obsspace(env))

	println("Creating policy")
	p = Policy(nn)  # TODO seed policy creation
	p.θ = bcast(p.θ, comm)
	println("policy created")
	obstat = Obstat(obssize, 1f-2)
	nt, win = NoiseTable(nt_size, length(p.θ), σ, comm)
	println("nt created")
	opt = isroot(comm) ? Adam(length(p.θ), η) : nothing

	f = (nn; show_dist = false) -> eval_net(nn, env, mean(obstat), std(obstat); show_dist=show_dist)
	tot_steps = 0

	for i in 1:gens
		if isroot(comm) println("\n\nGen $i") end
		
		t = now()
		obstat += step(p, nt, f, episodes, opt, comm)  # TODO pass through total steps
		
		if isroot(comm) 
			@show now() - t
			@show first(f(to_nn(p); show_dist=true)) 
		end
		
	end

	MPI.free(win)
end

function eval_net(nn::Chain, env, obmean, obstd; show_dist=false)
	reset!(env)
	obs = []

	r = 0
	step = 0

	for i in 1:500
		ob = getobs(env)
		act = forward(nn, ob, obmean, obstd)
		# if !all(isfinite.(act)) @show act "NOT FINITE!"; end
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

function step(π::AbstractPolicy, nt, f, n::Int, optim, comm::Comm; l2coeff=0.005f0)  # TODO rename this because it mutates π
	local_results, (s, ssq, c) = evaluate(π, nt, f, n ÷ size(comm))  # todo check that n is divisible by size
	# @show SVector{length(s), Float32}(s), SVector{length(s), Float32}(ssq), c
	local_obstat = Obstat{length(s), Float32}(SVector{length(s), Float32}(s), SVector{length(s), Float32}(ssq), c)
	results = gather(local_results, comm)
	obstat = allreduce(local_obstat, +, comm)

	if isroot(comm)
		ranked = rank(results)
		# TODO clean this up - minus positive noise fit from neg and adding up steps
		ranked = map((r) -> EsResult(first(r).fit - last(r).fit, first(r).ind, first(r).steps + last(r).steps), partition(ranked, 2))
		grad = l2coeff * π.θ - approxgrad(nt, ranked) ./ (n * 2)
		optimize!(π, optim, grad) # if this returns a new policy then Policy can be immutable
		
		@show Util.summary(results)
	end

	MPI.Barrier(comm)
	π.θ = bcast(π.θ, comm)
	# @show obstat

	obstat  # don't like passing this out, but think it is neccesary for `f`
end

function eval_one(pol::AbstractPolicy, nt::NoiseTable, noise_ind, f)
	pπ, nπ, noise_ind = noiseify(pol, nt, noise_ind)  # todo make this

	pfit, psteps, pobs = f(to_nn(pπ))
	nfit, nsteps, nobs = f(to_nn(nπ))

	# These are vecs, how to not pass them back to master?
	sm, sumsq, cnt = zeros(size(first(pobs))), zeros(size(first(pobs))), 0  # svector from the begining
	if rand() < 0.01
		sm = sum(vcat(pobs, nobs))
		sumsq = sum(map(x -> x.^2, vcat(pobs, nobs)))
		cnt = length(pobs) + length(nobs)
		# all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, possibly less compute?
	end

	EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps), (sm, sumsq, cnt)
end

function evaluate(pol::AbstractPolicy, nt, f, n::Int)
	# TODO store fits as Float32
	results = Vector{EsResult{Float64}}()  # [positive EsResult 1, negative EsResult 1, ...]
	sm, sumsq, cnt = [], [], 0

	for i in 1:n
		pπ, nπ, noise_ind = noiseify(pol, nt)

		pfit, psteps, pobs = f(to_nn(pπ))
		nfit, nsteps, nobs = f(to_nn(nπ))

		if i == 1 sm, sumsq = zeros(size(first(pobs))), zeros(size(first(pobs))) end

		if rand() < 0.01
			sm .+= sum(vcat(pobs, nobs))
			sumsq .+= sum(map(x -> x.^2, vcat(pobs, nobs)))
			cnt += length(pobs) + length(nobs)
			# all_obs = vcat(all_obs, pobs, nobs)  # lots of mem, possibly less compute?
		end
		push!(results, EsResult(pfit, noise_ind, psteps), EsResult(nfit, noise_ind, nsteps))
	end

	results, (sm, sumsq, cnt)
end

noiseify(pol::Policy, nt::NoiseTable) = noiseify(pol, nt, rand_ind(nt))

function noiseify(pol::Policy, nt::NoiseTable, ind::Int)
	noise = sample(nt, ind)
	Policy(pol.θ .+ noise, pol._nn_maker), Policy(pol.θ .- noise, pol._nn_maker), ind
end

function approxgrad(nt::NoiseTable, rankedresults)
	fits = map(r -> r.fit, rankedresults)
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

	# if !all(isfinite.(out)) 
	# 	@show obmean obstd
	# end


	out .+ r
end

end
