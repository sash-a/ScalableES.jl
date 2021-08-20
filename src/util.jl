module Util

export EsResult, rank

import StatsBase

struct EsResult{T}
	fit::T
	ind::Int
	steps::Int
end

function rank(fs::Vector{T}) where T
	len = length(fs)
	inds = sortperm(fs)
	ranked = zeros(Float32, len)
	ranked[inds] = (0:len - 1)

	ranked ./ (len - 1) .- 0.5f0
end

rank(results::Vector{EsResult{T}}) where T = map((r,f)->EsResult(f, r.ind, r.steps), results, rank((r->r.fit).(results)))

function StatsBase.describe(results::Vector{EsResult{T}}) where T
	fits = map(r->r.fit, results)
	StatsBase.describe(fits)
end

end