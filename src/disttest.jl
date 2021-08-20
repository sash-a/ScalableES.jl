include("mpiutils.jl")
include("vbn.jl")

using .MpiUtils
using .Vbn
using StaticArrays
import MPI

MPI.Init()

comm = MPI.COMM_WORLD
root = 0

sz =  MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# @show rank, sz

# node_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, 0)

# struct Immut{T}
#     r::T
#     s::Int
# end

# Immut{Float32}(32f0, 1)
# Immut(32f0, 1)

# i = Immut(rank, size)
# buf = MPI.UBuffer(fill(i, size - 1), size - 1)

# recv = MPI.Alltoall(buf, comm)

# recv = MPI.Gather(fill(i, 5), root, comm)

# SArray
sz = 2 
s = SVector{sz, Float32}(fill(1 * 1f0, sz))
ssq = SVector{sz, Float32}(fill(2 * 1f0, sz))
o = Obstat{sz, Float32}(s, ssq, 1)
o2 = Obstat(sz, 2.f0)

o2 += o
# reduced=allreduce(o, +, comm)


# include("policy.jl")
# using .Plcy
# using Flux

# p = Policy(Dense(5,5))
# @show p.θ
# # scattered = scatter(p.θ, comm)
 
# # recv = MPI.bcast(p.θ, root, comm)
# include("util.jl")
# using .Util
# recv=gather([EsResult{Int}(rank, 1, 1), EsResult(rank, 2,2), EsResult(rank, 3,3)], comm)


@show rank, recv
