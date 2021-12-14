abstract type AbstractOptim end
function optimize(::AbstractOptim, grad) end

mutable struct Adam <: AbstractOptim
    dim::Int
    lr::Real
    t::Int
    beta1::Real
    beta2::Real
    epsilon::Real
    m::Vector
    v::Vector

    Adam(dim::Int, lr::Real) = new(dim, lr, 0, 0.9, 0.999, 1e-08, zeros(Float32, dim), zeros(Float32, dim))
end

function optimize(opt::Adam, grad)
    opt.t += 1

    a = opt.lr * sqrt(1 - opt.beta2^opt.t) / (1 - opt.beta1^opt.t)
    opt.m = opt.beta1 * opt.m + (1 - opt.beta1) * grad
    opt.v = opt.beta2 * opt.v + (1 - opt.beta2) * (grad .^ 2)

    return -a * opt.m ./ (sqrt.(opt.v) .+ opt.epsilon)
end

function optimize!(
    π::Policy,
    ranked::Vector{EsResult{T}},
    nt::NoiseTable,
    optim::AbstractOptim,
    l2coeff::Float32,
) where {T<:AbstractFloat}
    grad = l2coeff * π.θ - approxgrad(π, nt, ranked) ./ (length(ranked) * 2)
    π.θ .+= optimize(optim, grad)
end
