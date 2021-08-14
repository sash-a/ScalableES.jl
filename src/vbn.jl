module Vbn

export Obstat, inc, mean, std

import Statistics: mean, std

struct Obstat
    sum
    sumsq
    count
end
Obstat(shape, eps) = Obstat(zeros(Float32, shape), fill(eps, shape), eps)

inc(o::Obstat, sum, sumsq, count) = Obstat(o.sum+sum, o.sumsq+sumsq, o.count+count)
mean(o::Obstat) = o.sum / o.count
std(o::Obstat) = sqrt.(max.(o.sumsq ./ o.count .- mean(o) .^ 2, fill(1f-2, size(o.sum))))

end  # module