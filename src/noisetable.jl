module Noise
export NoiseTable, sample, rand_ind, make_noise

using Distributions

struct NoiseTable
	noise::Vector
	noise_len::Int
    σ::Real
end

NoiseTable(table_size::Int, noise_len::Int, σ::Float32) = NoiseTable(make_noise(table_size, σ), noise_len, σ)

make_noise(table_size::Int, σ::Float32) = rand(Normal{Float32}(0f0, σ), table_size)

rand_ind(nt::NoiseTable) = rand(1:length(nt.noise) - nt.noise_len)
rand_ind(noise::Vector, noise_len::Int) = rand(1:length(noise) - noise_len)

sample(noise::Vector, pos::Int, noise_len::Int) = noise[pos:pos+noise_len - 1]
function sample(noise::Vector, noise_len::Int)
    i = rand_ind(noise, noise_len)
    sample(noise, i, noise_len), i
end

sample(nt::NoiseTable, pos::Int) = sample(nt.noise, pos, nt.noise_len)
sample(nt::NoiseTable) = sample(nt.noise, nt.noise_len)

end  # module