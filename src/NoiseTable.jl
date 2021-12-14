struct NoiseTable{T<:AbstractFloat}
	noise::AbstractVector{T}
	noise_len::Int
    σ::T
end

NoiseTable(table_size::Int, noise_len::Int, σ::Float32) = NoiseTable{Float32}(rand(Normal{Float32}(0f0, σ), table_size), noise_len, σ)

# TODO generate and share seed
function NoiseTable(table_size::Int, noise_len::Int, σ::Float32, ::AbstractComm; seed=123)
    rng = MersenneTwister(seed)
    noise = rand(rng, Normal{Float32}(0f0, σ), table_size)
    NoiseTable{Float32}(noise, noise_len, σ), nothing
end

# Shared array using mpi shared win
# function NoiseTable(table_size::Int, noise_len::Int, σ::Float32, comm::MPI.Comm)
#     win, shared_arr = mpi_shared_array(comm, Float32, (table_size,))
#     if isroot(comm)
#         shared_arr[:] = rand(Normal{Float32}(0f0, σ), table_size)
#     end

#     MPI.Barrier(comm) # finish writing before reading
#     NoiseTable(shared_arr, noise_len, σ), win  # so that win can be freed
# end

Base.rand(rng::AbstractRNG, nt::NoiseTable) = rand(rng, nt, nt.noise_len)
Base.rand(rng::AbstractRNG, nt::NoiseTable, len::Int) = rand(rng, 1:length(nt.noise) - len)

@inbounds StatsBase.sample(nt::NoiseTable, pos::Int, len::Int) = nt.noise[pos:pos + len - 1]
StatsBase.sample(nt::NoiseTable, pos::Int) = StatsBase.sample(nt, pos, nt.noise_len)
function StatsBase.sample(rng::AbstractRNG, nt::NoiseTable)
    i = rand(rng, nt)
    StatsBase.sample(nt, i), i
end