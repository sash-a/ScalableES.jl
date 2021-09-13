import Base: +

abstract type AbstractObstat end

struct Obstat{S, T<:AbstractFloat} <: AbstractObstat
    sum::SArray{Tuple{S}, T, 1, S}
    sumsq::SArray{Tuple{S}, T, 1, S}
    count::T
end
Obstat(shape, eps::Float32) = Obstat{shape, Float32}(SVector{shape, Float32}(zeros(Float32, shape)), SVector{shape, Float32}(fill(eps, shape)), eps)

function add_obs(obstat::Obstat{S, T}, obs) where S where T <: AbstractFloat
    if isempty(obs) return obstat end

    shape = length(obstat.sum)
    sm = obstat.sum .+ SVector{shape, Float32}(sum(obs))
    ssq = obstat.sumsq .+ SVector{shape, Float32}(sum(map(x -> x.^2, obs)))
    Obstat(sm, ssq, obstat.count + length(obs))
end

+(x::Obstat, y::Obstat) = Obstat(x.sum .+ y.sum, x.sumsq .+ y.sumsq, x.count + y.count)
inc(o::Obstat, sum, sumsq, count) = o + Obstat(sum, sumsq, count)
mean(o::Obstat) = o.sum / o.count
std(o::Obstat) = sqrt.(max.(o.sumsq ./ o.count .- mean(o) .^ 2, fill(1f-2, size(o.sum))))