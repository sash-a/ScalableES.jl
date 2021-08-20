module Noise

include("mpiutils.jl")
using .MpiUtils

using MPI
using Distributions
import StatsBase

export NoiseTable, sample, rand_ind

struct NoiseTable
	noise::Array
	noise_len::Int
    σ::Real
end

# NoiseTable(table_size::Int, noise_len::Int, σ::Real) = NoiseTable(randn(Float32, table_size) * σ, noise_len, σ)
NoiseTable(table_size::Int, noise_len::Int, σ::Float32) = NoiseTable(rand(Normal{Float32}(0f0, σ), table_size), noise_len, σ)
function NoiseTable(table_size::Int, noise_len::Int, σ::Float32, comm::MPI.Comm)
    win, shared_arr = mpi_shared_array(comm, Float32, (table_size,))
    if isroot(comm)
        shared_arr[:] = rand(Normal{Float32}(0f0, σ), table_size)
    end
    
    NoiseTable(shared_arr, noise_len, σ), win  # so that win can be freed
end

rand_ind(nt::NoiseTable) = rand(1:length(nt.noise) - nt.noise_len)

StatsBase.sample(nt::NoiseTable, pos::Int) = nt.noise[pos:pos + nt.noise_len - 1]
function StatsBase.sample(nt::NoiseTable)
    i = rand_ind(nt)
    StatsBase.sample(nt, i), i
end

end