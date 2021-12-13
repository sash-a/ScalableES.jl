module ScalableES

using MPI: MPI, Comm
using Base.Threads
using ThreadPools
using SharedArrays

using LyceumMuJoCo
using MuJoCo
using HrlMuJoCoEnvs

using Flux
using StaticArrays
using Distributions
using StatsBase
import Statistics: mean, std

using Dates
using BSON: @save
using TensorBoardLogger, Logging
using IterTools
using Distances
using Random
using Future

include("MpiUtils.jl")
include("Policy.jl")
include("NoiseTable.jl")
include("Optim.jl")
include("Obstat.jl")
include("Util.jl")

export run_es, run_nses

function run_es(name::String, nn, envs, comm::AbstractComm; 
				gens=150, npolicies=256, steps=500, episodes=3, σ=0.02f0, nt_size=250000000, η=0.01f0)
	@assert npolicies / nnodes(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(nnodes(comm)))"

	println("Running ScalableEs")
	if isroot(comm)
		tblg = TBLogger("tensorboard_logs/$(name)", min_level=Logging.Info)
	end

	env = first(envs)
	obssize = length(obsspace(env))

	println("Creating policy")
	p = ScalableES.Policy(nn, comm)
	bcast_policy!(p, comm)

	println("Creating rngs")
	rngs = parallel_rngs(123, nprocs(comm), comm)

	println("Creating noise table")
	nt, win = NoiseTable(nt_size, length(p.θ), σ, comm)
	
	obstat = Obstat(obssize, 1f-2)
	opt = isroot(comm) ? Adam(length(p.θ), η) : nothing

	println("Initialization done")
	f = (nn, e, rng, obmean, obstd) -> eval_net(nn, e, obmean, obstd, steps, episodes, rng)
	evalfn = (nn, e, rng, obmean, obstd) -> first(eval_net(nn, e, obmean, obstd, steps, episodes, rng))

	f(to_nn(p), first(envs), first(rngs), mean(obstat), std(obstat))

	t = time_ns()
	run_gens(gens, name, p, nt, f, evalfn, envs, npolicies, opt, obstat, tblg, rngs, comm)
    println("Total time: $((time_ns() - t) / 1e9) s")

	@save joinpath("saved", name, "policy-obstat-opt-final.bson") p obstat opt

	if win !== nothing
		MPI.free(win)
	end
end


function run_gens(n::Int, 
				  name::String,
				  p::AbstractPolicy, 
		 		  nt::NoiseTable, 
  				  fn,
				  eval_fn, 
				  envs, 
				  npolicies::Int, 
				  opt::Union{AbstractOptim, Nothing}, 
				  obstat::AbstractObstat, 
				  logger,
				  rngs,
				  comm::Union{Comm, ThreadComm})
	# tot_steps = 0
	# eval_score = -Inf
	# env = first(envs)

	for i in 1:n
		t = time_ns()
		# I dislike this and would rather f is simply passed to this function, 
		#  but that doesn't allow for obmean and obstd to be updated in the 
		#  partial function f
		@time f = (nn, e, rng) -> fn(nn, e, rng, mean(obstat), std(obstat))
		@time res, gen_obstat = step_es(p, nt, f, envs, npolicies, opt, rngs, comm)
		@time obstat += gen_obstat
		# gt = (time_ns() - t) / 1e9

		if isroot(comm)
			println("\n\nGen $i")

			# gen_eval = checkpoint(i, name, p, obstat, opt, eval_fn, env, eval_score, first(rngs))
			# eval_score = max(eval_score, gen_eval)
			
			# tot_steps += sumsteps(res)
			# loginfo(logger, gen_eval, res, tot_steps, gt)
		end
	end
end

function eval_net(nn::Chain, env, obmean, obstd, steps::Int, episodes::Int, rng)
	obs = Vector{Vector{Float64}}()
	r = 0.
	step = 0

	for e in 1:episodes
		LyceumMuJoCo.reset!(env)
		for i in 1:steps
			ob = getobs(env)
			act = forward(nn, ob, obmean, obstd, rng)
			setaction!(env, act)
			step!(env)

			step += 1
			# push!(obs, ob)  # propogate ob recording to here, don't have to alloc mem if not using obs
			r += getreward(env)
			# if isdone(env) break end
		end
	end
	# @show rew step
	r / episodes, step, obs
end

function step_es(π::AbstractPolicy, nt, f, envs, n::Int, optim, rngs, comm::AbstractComm; l2coeff=0.005f0)  # TODO rename this because it mutates π
	st = now()
	# println("[$(procrank(comm))] Evaluating $(n ÷ nnodes(comm)) policies")
	eval_alloc = @allocated local_results, obstat = evaluate(π, nt, f, envs, n ÷ nnodes(comm), rngs, comm)
	# println("[$(procrank(comm))] Eval time: $(now() - st)")
	t = now()
	results, obstat = share_results(local_results, obstat, comm)

	# println("[$(procrank(comm))] Res share time: $(now() - t)")


	if isroot(comm)
		# println("[$(procrank(comm))] Eval + share time: $(now() - st)")
		t = now()
		ranked = rank(results)
		optimize!(π, ranked, nt, optim, l2coeff)  # if this returns a new policy then Policy can be immutable
		# println("[$(procrank(comm))] Opt time: $(now() - t)")
	end

	t = now()
	bcast_policy!(π, comm)

	# println("[$(procrank(comm))] Pol share time: $(now() - t)")
	# println("[$(procrank(comm))] Total time: $(now() - st)")

	results, obstat
end


"""
Results and obstat are empty containers of the correct type
"""
function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int, results, obstat, rngs, comm)
	l = ReentrantLock()

	@qthreads for i in 1:n
		env = envs[Threads.threadid()]
		rng = rngs[Threads.threadid()]

		t = now()
		pπ, nπ, noise_ind = noiseify(pol, nt, rng)
		# println("[$(procrank(comm))] noiseify time: $(now() - t)")

		t = now()
		pnn = to_nn(pπ)
		nnn = to_nn(nπ)

		# println("[$(procrank(comm))] to_nn time: $(now() - t)")

		t = now()
		pfit, psteps, pobs = f(pnn, env, rng)
		nfit, nsteps, nobs = f(nnn, env, rng)
		# println("[$(procrank(comm))] env runtime: $(now() - t)")

		# if rand(rng) < 0.001
		# 	t = now()
		# 	Base.@lock l begin
		# 		# obstat = add_obs(obstat, vcat(pobs, nobs))
		# 		obstat = add_obs(obstat, pobs)
		# 		obstat = add_obs(obstat, nobs)
		# 	end
		# 	# println("[$(procrank(comm))] add obstat time: $(now() - t)")
		# end

		t = now()
		@inbounds results[i * 2 - 1] = make_result(pfit, noise_ind, psteps)
		@inbounds results[i * 2] = make_result(nfit, noise_ind, nsteps)
		# println("[$(procrank(comm))] make res time: $(now() - t)")
	end

	results, obstat
end

function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int, rngs, comm::AbstractComm)
	# TODO store fits as Float32
	results = make_result_vec(n, pol, comm)  # [positive EsResult 1, negative EsResult 1, ...]
	obstat = make_obstat(length(obsspace(first(envs))), pol)

	evaluate(pol, nt, f, envs, n ÷ 2, results, obstat, rngs, comm)  # ÷ 2 because sampling pos and neg
end

# Policy methods
noiseify(pol::AbstractPolicy, nt::NoiseTable, rng) = noiseify(pol, nt, rand(rng, nt))
function noiseify(pol::Policy, nt::NoiseTable, ind::Int)
	noise = sample(nt, ind)
	Policy(pol.θ .+ noise, pol._re), Policy(pol.θ .- noise, pol._re), ind
end

function approxgrad(π::AbstractPolicy, nt::NoiseTable, rankedresults::Vector{EsResult{T}}) where T <: AbstractFloat
	fits = map(r -> r.fit, rankedresults)
	noises = map(r -> sample(nt, r.ind, length(π.θ)), rankedresults)
		
	sum([f .* n for (f, n) in zip(fits, noises)]) .* (1 / nt.σ)  # noise already has std σ, which just messes with lr 
end

function optimize!(π::Policy, ranked::Vector{EsResult{T}}, nt::NoiseTable, optim::AbstractOptim, l2coeff::Float32) where T <: AbstractFloat
	grad = l2coeff * π.θ - approxgrad(π, nt, ranked) ./ (length(ranked) * 2)
	π.θ .+= optimize(optim, grad)
end

# make_result_vec(n::Int, ::Policy, ::Comm) = Vector{EsResult{Float64}}(undef, n)
make_result_vec(n::Int, ::Policy, ::AbstractComm) = Vector{EsResult{Float64}}(undef, n)#SharedVector{EsResult{Float64}}(n)
make_obstat(shape, ::Policy) = Obstat(shape, 0f0)

# MPI stuff
function share_results(local_results::AbstractVector{T}, local_obstat::S, comm::Comm) where S <: AbstractObstat where T <: AbstractResult
	# local_obstat = Obstat{length(sum), Float32}(SVector{length(sum), Float32}(sum), SVector{length(sum), Float32}(ssq), cnt)
	results = gather(local_results, comm)
	obstat = allreduce(local_obstat, +, comm)

	results, obstat
end
function share_results(local_results::AbstractVector{T}, local_obstat::S, ::ThreadComm) where S <: AbstractObstat where T <: AbstractResult
	local_results, local_obstat  # no need to do any sharing if not using mpi
end

function bcast_policy!(::AbstractPolicy, ::ThreadComm) end  # no need to do any sharing if not using mpi
bcast_policy!(π::Policy, comm::Comm) = π.θ[:] = bcast(π.θ, comm)

# nn methods
function forward(nn, x, obmean, obstd, rng)
	x = clamp.((x .- obmean) ./ obstd, -5, 5)
	out = nn(x)
	
	if rng !== nothing
		r = randn(rng, Float32, size(out)) .* 0.01
		out .+ r
	else
		out
	end
end

include("novelty/ScalableNsEs.jl")

end  # module
