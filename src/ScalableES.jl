module ScalableES

using MPI: MPI, Comm
using Base.Threads
using SharedArrays

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

function run_es(name::String, nn, envs, comm::Union{Comm, ThreadComm}; 
				gens=150, npolicies=256, steps=500, episodes=3, σ=0.02f0, nt_size=250000000, η=0.01f0)
	@assert npolicies / size(comm) % 2 == 0 "Num policies / num nodes must be even (eps:$npolicies, nprocs:$(size(comm)))"

	println("Running ScalableEs")
	tblg = TBLogger("tensorboard_logs/$(name)", min_level=Logging.Info)

	env = first(envs)
	obssize = length(obsspace(env))

	println("Creating policy")
	p = ScalableES.Policy(nn, comm)
	bcast_policy!(p, comm)

	println("Creating noise table")
	nt, win = NoiseTable(nt_size, length(p.θ), σ, comm)
	
	obstat = Obstat(obssize, 1f-2)
	opt = isroot(comm) ? Adam(length(p.θ), η) : nothing

	println("Initialization done")
	f = (nn, e, obmean, obstd) -> eval_net(nn, e, obmean, obstd, steps, episodes)
	run_gens(gens, name, p, nt, f, envs, npolicies, opt, obstat, tblg, comm)

	model = to_nn(p)
	@save joinpath("saved", name, "model-obstat-opt-final.bson") model obstat opt

	if win !== nothing
		MPI.free(win)
	end
end


function run_gens(n::Int, 
				  name::String,
				  p::AbstractPolicy, 
		 		  nt::NoiseTable, 
  				  fn, 
				  envs, 
				  npolicies::Int, 
				  opt::AbstractOptim, 
				  obstat::AbstractObstat, 
				  logger,
				  comm::Union{Comm, ThreadComm})
	tot_steps = 0
	eval_score = -Inf
	env = first(envs)

	for i in 1:n		
		t = now()
		# I dislike this and would rather f is simply passed to this function, 
		#  but that doesn't allow for obmean and obstd to be updated in the 
		#  partial function f
		f = (nn, e) -> fn(nn, e, mean(obstat), std(obstat))
		res, gen_obstat = step_es(p, nt, f, envs, npolicies, opt, comm)
		obstat += gen_obstat

		if isroot(comm) 
			println("\n\nGen $i")
			tot_steps += sumsteps(res)

			# save model
			gen_eval = geteval(env)
			model = to_nn(p)
			if gen_eval > eval_score || i % 10 == 0
				println("Saving model with eval score $gen_eval")
				path = joinpath("saved", name, "model-obstat-opt-gen$i.bson")
				@save path model obstat opt
			end
			eval_score = max(eval_score, gen_eval)

			loginfo(logger, first(f(model, env)), res, tot_steps, t)
		end
	end
end

function eval_net(nn::Chain, env, obmean, obstd, steps::Int, episodes::Int)
	obs = Vector{Vector{Float64}}()
	r = 0.
	step = 0

	for i in 1:episodes
		LyceumMuJoCo.reset!(env)
		for i in 1:steps
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
	r / episodes, step, obs
end

function step_es(π::AbstractPolicy, nt, f, envs, n::Int, optim, comm::Union{Comm, ThreadComm}; l2coeff=0.005f0)  # TODO rename this because it mutates π
	local_results, obstat = evaluate(π, nt, f, envs, n ÷ size(comm) ÷ 2, comm)
	results, obstat = share_results(local_results, obstat, comm)

	if isroot(comm)
		ranked = rank(results)
		optimize!(π, ranked, nt, optim, l2coeff)  # if this returns a new policy then Policy can be immutable
	end

	bcast_policy!(π, comm)

	results, obstat
end

function evaluate(pol::AbstractPolicy, nt, f, envs, n::Int, comm::Union{Comm, ThreadComm})
	# TODO store fits as Float32
	results = make_result_vec(n * 2, pol, comm)  # [positive EsResult 1, negative EsResult 1, ...]
	obstat = make_obstat(length(obsspace(first(envs))), pol)

	l = ReentrantLock()

	Threads.@threads for i in 1:n
		env = envs[Threads.threadid()]

		pπ, nπ, noise_ind = noiseify(pol, nt)

		pfit, psteps, pobs = f(to_nn(pπ), env)
		nfit, nsteps, nobs = f(to_nn(nπ), env)

		if rand() < 0.01
			Base.@lock l begin
				# obstat = add_obs(obstat, vcat(pobs, nobs))
				# TODO shared arrays here also?
				obstat = add_obs(obstat, pobs)
				obstat = add_obs(obstat, nobs)
			end
		end

		results[i * 2 - 1] = make_result(pfit, noise_ind, psteps)
		results[i * 2] = make_result(nfit, noise_ind, nsteps)
	end

	results, obstat
end

# Policy methods
noiseify(pol::Policy, nt::NoiseTable) = noiseify(pol, nt, rand(nt))
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

make_result_vec(n::Int, ::Policy, ::Comm) = Vector{EsResult{Float64}}(undef, n)
make_result_vec(n::Int, ::Policy, ::ThreadComm) = SharedVector{EsResult{Float64}}(n)

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


function bcast_policy!(π::Policy, comm::Comm)
	MPI.Barrier(comm)
	π.θ = bcast(π.θ, comm)
end

function bcast_policy!(::Policy, ::ThreadComm) end # no need to do any sharing if not using mpi

# nn methods
function forward(nn, x, obmean, obstd; rng=Random.GLOBAL_RNG)
	x = clamp.((x .- obmean) ./ obstd, -5, 5)
	out = nn(x)
	
	r = zeros(size(out))
	if rng !== nothing
		r = randn(rng, Float32, size(out)) .* 0.01
	end

	out .+ r
end

end  # module
