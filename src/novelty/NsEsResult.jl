struct NsEsResult{T,B,S} <: AbstractResult{T}
    behaviours::SVector{B,SPath{S}}
    novelty::T
    result::EsResult{T}
end

function ScalableES.make_result(fit::Tuple{Float64,Vector{Path}}, noise_ind::Int, steps::Int)
    n_behvs = length(last(fit))
    behv_size = length(last(last(fit)))
    static_paths = map(p -> SPath{behv_size}(SVector{behv_size,Vec2}(p)), last(fit))  # converting the multiple path vectors to svectors
    static_paths = SVector{n_behvs,SPath{behv_size}}(static_paths)  # converting vector of paths to an svector
    NsEsResult{Float64,n_behvs,behv_size}(static_paths, -1., EsResult{Float64}(first(fit), noise_ind, steps))
end

function ScalableES.make_result_vec(n::Int, ::Policy, rollouts::Int, steps::Int, interval::Int, ::ThreadComm)
    SharedVector{NsEsResult{Float64,rollouts,steps ÷ interval}}(n)
end

meanfit(rs::AbstractVector{T}) where T <: NsEsResult = mean(map(r->r.result.fit, rs))

"""
Rank novelty results by shaping the novelty and fitness of each policy separately 
then weighting them by `w` and `1 - w` and adding corresping weights and novelties
"""
function ScalableES.rank(rs::AbstractVector{T}, w) where T <: NsEsResult
    shaped_novelties = map(((p, n),) -> p - n, partition(ScalableES.rank(map(r->r.novelty, rs)), 2))
    shaped_fits = ScalableES.rank(map(r -> r.result, rs))

    # shaped_fits * w + shaped_novelties * (1 - w)  # <- might be more efficient? But harder to pack back into EsResults
    map((nov, res) -> EsResult(res.fit * w + nov * (1 - w), res.ind, res.steps), shaped_novelties, shaped_fits)
end

function ScalableES.sumsteps(res::AbstractVector{T}) where T <: NsEsResult
    sumsteps(map(r -> r.result, res))
end

function ScalableES.loginfo(tblogger, 
                            main_fit, 
                            rs::Vector{T},
                            tot_steps::Int, 
                            start_time,
                            w,
                            tsb_fit) where T <: NsEsResult

    fitstats = summarystats(map(r->r.result, rs))
    novstats = summarystats(map(r->r.novelty, rs))
    
	println("Main fit: $main_fit")
    println("Time since best fit: $tsb_fit")
    println("Fitness weight: $w")
	println("Total steps: $tot_steps")
	println("Time: $(now() - start_time)")
	println("Fitness stats:\n$fitstats")
	println("Novelty stats:\n$novstats")
    println("---------------------------------------------")

	with_logger(tblogger) do
		@info "" main_fitness=main_fit log_step_increment=0
        @info "" tsb_fit=tsb_fit log_step_increment=0
        @info "" fit_w=w log_step_increment=0
		@info "" fitstats=fitstats log_step_increment=0
		@info "" summarystat=fitstats log_step_increment=0
		@info "" novstats=novstats log_step_increment=0
		@info "" total_steps=tot_steps log_step_increment=1
	end
end