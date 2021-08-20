module Vbn

export Obstat, inc, mean, std

import Statistics: mean, std
import Base: +
using StaticArrays

struct Obstat{S, T<:AbstractFloat}
    sum::SArray{Tuple{S}, T, 1, S}
    sumsq::SArray{Tuple{S}, T, 1, S}
    count::T
end
Obstat(shape, eps::Float32) = Obstat{shape, Float32}(SVector{shape, Float32}(zeros(Float32, shape)), SVector{shape, Float32}(fill(eps, shape)), eps)

+(x::Obstat, y::Obstat) = Obstat(x.sum .+ y.sum, x.sumsq .+ y.sumsq, x.count + y.count)
inc(o::Obstat, sum, sumsq, count) = o + Obstat(sum, sumsq, count)
mean(o::Obstat) = o.sum / o.count
std(o::Obstat) = sqrt.(max.(o.sumsq ./ o.count .- mean(o) .^ 2, fill(1f-2, size(o.sum))))

end  # module