module ScalableES

using MPI: MPI, Comm
using Base.Threads

using LyceumMuJoCo
using MuJoCo

using Flux
using Random
using IterTools
using StaticArrays
using Distributions
using StatsBase
import Statistics: mean, std

using Dates
using BSON: @save
using TensorBoardLogger, Logging

include("MpiUtils.jl")
include("Policy.jl")
include("NoiseTable.jl")
include("Optim.jl")
include("Obstat.jl")
include("Util.jl")


export run_es

function run_es(name::String, nn, envs, comm::Comm; 
				gens=150, npolicies=256, episodes=3, σ=0.02f0, nt_size=250000000, η=0.01f0)
	@assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

	println("Running ScalableEs")
	tblg = TBLogger("tensorboard_logs/$(name)", min_level=Logging.Info)

	env = first(envs)
	obssize = length(obsspace(env))

	println("Creating policy")
	p = ScalableES.Policy(nn)
	p.θ = bcast(p.θ, comm)

	println("Creating noise table")
	nt, win = NoiseTable(nt_size, length(p.θ), σ, comm)
	
	obstat = Obstat(obssize, 1f-2)
	opt = isroot(comm) ? Adam(length(p.θ), η) : nothing
	f = (nn, e) -> eval_net(nn, e, mean(obstat), std(obstat), episodes)


	println("Initialization done")
	run_gens(gens, name, p, nt, f, envs, npolicies, opt, obstat, tblg, comm)

	model = to_nn(p)
	@save joinpath("saved", name, "model-obstat-opt-final.bson") model obstat opt

	MPI.free(win)
end


function run_gens(n::Int, 
				  name::String,
				  p::AbstractPolicy, 
		 		  nt::NoiseTable, 
  				  f, 
				  envs, 
				  npolicies::Int, 
				  opt::AbstractOptim, 
				  obstat::Obstat, 
				  logger,
				  comm::Comm)
	tot_steps = 0
	eval_score = -Inf
	env = first(envs)

	for i in 1:n		
		t = now()
		res, gen_obstat = step_es(p, nt, f, envs, npolicies, opt, comm)
		obstat += gen_obstat

		if isroot(comm) 
			println("\n\nGen $i")
			tot_steps += sum(map(r -> r.steps, res))

			# save model
			gen_eval = geteval(env)
			if gen_eval > eval_score || i % 10 == 0
				println("Saving model with eval score $gen_eval")
				eval_score = gen_eval
				model = to_nn(p)
				path = joinpath("saved", name, "model-obstat-opt-gen$i.bson")
				@save path model obstat opt
			end

			# print info
			fit = first(f(to_nn(p), env))
			ss = summarystats(res)
			println("Main fit: $(fit)")
			println("Total steps: $tot_steps")
			println("Time: $(now() - t)")
			println(ss)
			with_logger(logger) do
				@info "" main_fitness=fit log_step_increment=0
				@info "" summarystat=ss log_step_increment=0
				@info "" total_steps=tot_steps log_step_increment=1
			end
		end
	end
end

function eval_net(nn::Chain, env, obmean, obstd, episodes::Int)
	obs = []
	r = 0
	step = 0

	for i in 1:episodes
		LyceumMuJoCo.reset!(env)
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
	end
	# @show rew step
	r/episodes, step, obs
end

function step_es(π::AbstractPolicy, nt, f, envs, n::Int, optim, comm::Comm; l2coeff=0.005f0)  # TODO rename this because it mutates π
	local_results, (s, ssq, c) = evaluate(π, nt, f, envs, n ÷ size(comm) ÷ 2)
	local_obstat = Obstat{length(s), Float32}(SVector{length(s), Float32}(s), SVector{length(s), Float32}(ssq), c)
	results = gather(local_results, comm)
	obstat = allreduce(local_obstat, +, comm)

	if isroot(comm)
		ranked = rank(results)
		ranked = map(((p, n),) -> EsResult(p.fit - n.fit, p.ind, p.steps + n.steps), partition(ranked, 2))
		grad = l2coeff * π.θ - approxgrad(nt, ranked) ./ (n * 2)
		optimize!(π, optim, grad) # if this returns a new policy then Policy can be immutable
	end

	MPI.Barrier(comm)
	π.θ = bcast(π.θ, comm)

	results, obstat
end

function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int)
	# TODO store fits as Float32
	results = Vector{EsResult{Float64}}(undef, n * 2)  # [positive EsResult 1, negative EsResult 1, ...]

	osize = length(obsspace(first(envs)))
	sm, sumsq, cnt = zeros(osize), zeros(osize), 0

	l = ReentrantLock()

	Threads.@threads for i in 1:n
		env = envs[Threads.threadid()]

		pπ, nπ, noise_ind = noiseify(pol, nt)

		pfit, psteps, pobs = f(to_nn(pπ), env)
		nfit, nsteps, nobs = f(to_nn(nπ), env)

		if rand() < 0.01
			Base.@lock l begin
				sm .+= sum(vcat(pobs, nobs))
				sumsq .+= sum(map(x -> x.^2, vcat(pobs, nobs)))
				cnt += length(pobs) + length(nobs)
			end
		end

		results[i * 2 - 1] = EsResult(pfit, noise_ind, psteps)
		results[i * 2] = EsResult(nfit, noise_ind, nsteps)
	end

	results, (sm, sumsq, cnt)
end

noiseify(pol::Policy, nt::NoiseTable) = noiseify(pol, nt, rand(nt))
function noiseify(pol::Policy, nt::NoiseTable, ind::Int)
	noise = sample(nt, ind)
	Policy(pol.θ .+ noise, pol._re), Policy(pol.θ .- noise, pol._re), ind
end

function approxgrad(nt::NoiseTable, rankedresults)
	fits = map(r -> r.fit, rankedresults)
	noises = map(r -> sample(nt, r.ind), rankedresults)
		
	sum([f .* n for (f, n) in zip(fits, noises)]) .* (1 / nt.σ)  # noise already has std σ, which just messes with lr 
end

function optimize!(π::Policy, optim, grad)
	π.θ .+= optimize(optim, grad)
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
