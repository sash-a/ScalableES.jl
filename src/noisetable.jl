module Noise

export NoiseTable, get_noise

struct NoiseTable
	noise::Vector
	noise_len::Int
end
NoiseTable(table_size::Int, noise_len::Int) = NoiseTable(randn(Float32, table_size), noise_len)

get_noise(nt::NoiseTable, pos::Int) = nt.noise[pos:pos + nt.noise_len - 1]
function get_noise(nt::NoiseTable)
    i = rand(1:length(nt.noise)-nt.noise_len)
    get_noise(nt, i), i
end

end