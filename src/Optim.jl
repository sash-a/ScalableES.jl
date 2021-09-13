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

function optimize(opt::Adam, grad::Vector)
    opt.t += 1

    a = opt.lr * sqrt(1 - opt.beta2 ^ opt.t) / (1 - opt.beta1 ^ opt.t)
    opt.m = opt.beta1 * opt.m + (1 - opt.beta1) * grad
    opt.v = opt.beta2 * opt.v + (1 - opt.beta2) * (grad .^ 2)

    return -a * opt.m ./ (sqrt.(opt.v) .+ opt.epsilon)
end