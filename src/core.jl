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
using StatsBase
using Dates

function run(nn, env, comm::Comm, gens=100, episodes=256, σ=0.02f0, nt_size=250000000, η=0.01f0)
	@assert episodes / size(comm) % 2 == 0 "Num episodes / nprocs must be even (eps:$episodes, nprocs:$(size(comm)))"

	println("Running ScalableEs")

	obssize = length(obsspace(env))

	println("Creating policy")
	p = Policy(nn)
	p.θ = bcast(p.θ, comm)

	println("Creating noise table")
	nt, win = NoiseTable(nt_size, length(p.θ), σ, comm)
	
	obstat = Obstat(obssize, 1f-2)
	opt = isroot(comm) ? Adam(length(p.θ), η) : nothing
	f = (nn; show_dist = false) -> eval_net(nn, env, mean(obstat), std(obstat); show_dist=show_dist)
	tot_steps = 0

	println("Initialization done")

	for i in 1:gens
		if isroot(comm) println("\n\nGen $i") end
		
		t = now()
		res, gen_obstat = step(p, nt, f, episodes, opt, comm)  # TODO pass through total steps
		obstat += gen_obstat

		if isroot(comm) 
			tot_steps += sum(map(r -> r.steps, res))

			# print info
			println("Main fit: $(first(f(to_nn(p); show_dist=true)))")
			println("Time: $(now() - t)")
			describe(res)
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
	local_results, (s, ssq, c) = evaluate(π, nt, f, n ÷ size(comm) ÷ 2)
	local_obstat = Obstat{length(s), Float32}(SVector{length(s), Float32}(s), SVector{length(s), Float32}(ssq), c)
	results = gather(local_results, comm)
	obstat = allreduce(local_obstat, +, comm)

	if isroot(comm)
		ranked = rank(results)
		# TODO clean this up - minus positive noise fit from neg and adding up steps
		ranked = map((r) -> EsResult(first(r).fit - last(r).fit, first(r).ind, first(r).steps + last(r).steps), partition(ranked, 2))
		grad = l2coeff * π.θ - approxgrad(nt, ranked) ./ (n * 2)
		optimize!(π, optim, grad) # if this returns a new policy then Policy can be immutable
	end

	MPI.Barrier(comm)
	π.θ = bcast(π.θ, comm)

	results, obstat
end

function eval_one(pol::AbstractPolicy, nt::NoiseTable, noise_ind, f)
	pπ, nπ, noise_ind = noiseify(pol, nt, noise_ind)

	pfit, psteps, pobs = f(to_nn(pπ))
	nfit, nsteps, nobs = f(to_nn(nπ))

	sm, sumsq, cnt = zeros(size(first(pobs))), zeros(size(first(pobs))), 0  # TODO svector from the begining
	if rand() < 0.01
		sm = sum(vcat(pobs, nobs))
		sumsq = sum(map(x -> x.^2, vcat(pobs, nobs)))
		cnt = length(pobs) + length(nobs)
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
		
	sum([f .* n for (f, n) in zip(fits, noises)]) .* (1 / nt.σ)  # noise already has std σ, which just messes with lr 
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
