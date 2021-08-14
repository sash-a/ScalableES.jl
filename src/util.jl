module Util

export EsResult, rank, summary

struct EsResult
	fit::Real
	ind::Int
	steps::Int
end

function rank(fs::Vector)
	len = length(fs)
	inds = sortperm(fs)
	ranked = zeros(Float32, len)
	ranked[inds] = (0:len - 1)
	
	(ranked ./ (len - 1) .- 0.5f0) * 2
end

rank(results::Vector{EsResult}) = map((r,f)->EsResult(f, r.ind, r.steps), results, rank((r->r.fit).(results)))

function summary(results::Vector)
	fits = map(r->r.fit, results)
	avg_rew = reduce(+, fits)/length(results)  
	max_rew = max(fits...)
	tot_steps = reduce(+, map(r->r.steps, results))

	avg_rew, max_rew, tot_steps
end

end