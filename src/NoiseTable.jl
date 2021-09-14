struct NoiseTable{T<:AbstractFloat}
	noise::AbstractVector{T}
	noise_len::Int
    σ::T
end

# NoiseTable(table_size::Int, noise_len::Int, σ::Real) = NoiseTable(randn(Float32, table_size) * σ, noise_len, σ)
NoiseTable(table_size::Int, noise_len::Int, σ::Float32) = NoiseTable{Float32}(rand(Normal{Float32}(0f0, σ), table_size), noise_len, σ)

function NoiseTable(table_size::Int, noise_len::Int, σ::Float32, ::ThreadComm)
    noise = SharedVector{Float32}(table_size; init=s->s[localindices(s)] = rand(Normal{Float32}(0f0, σ), length(localindices(s))))
    NoiseTable{Float32}(noise, noise_len, σ), nothing
end

function NoiseTable(table_size::Int, noise_len::Int, σ::Float32, comm::MPI.Comm)
    win, shared_arr = mpi_shared_array(comm, Float32, (table_size,))
    if isroot(comm)
        shared_arr[:] = rand(Normal{Float32}(0f0, σ), table_size)
    end
    
    NoiseTable(shared_arr, noise_len, σ), win  # so that win can be freed
end

Base.rand(nt::NoiseTable) = rand(nt, nt.noise_len)
Base.rand(nt::NoiseTable, len::Int) = rand(1:length(nt.noise) - len)


StatsBase.sample(nt::NoiseTable, pos::Int, len::Int) = nt.noise[pos:pos + len - 1]
StatsBase.sample(nt::NoiseTable, pos::Int) = StatsBase.sample(nt, pos, nt.noise_len)
function StatsBase.sample(nt::NoiseTable)
    i = rand(nt)
    StatsBase.sample(nt, i), i
end