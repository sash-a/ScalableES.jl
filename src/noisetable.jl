module Noise

using Distributions

export NoiseTable, sample, rand_ind

struct NoiseTable
	noise::Vector
	noise_len::Int
    σ::Real
end

# NoiseTable(table_size::Int, noise_len::Int, σ::Real) = NoiseTable(randn(Float32, table_size) * σ, noise_len, σ)
NoiseTable(table_size::Int, noise_len::Int, σ::Float32) = NoiseTable(rand(Normal{Float32}(0f0, σ), table_size), noise_len, σ)

rand_ind(nt::NoiseTable) = rand(1:length(nt.noise) - nt.noise_len)

sample(nt::NoiseTable, pos::Int) = nt.noise[pos:pos + nt.noise_len - 1]
function sample(nt::NoiseTable)
    i = rand_ind(nt)
    sample(nt, i), i
end

end