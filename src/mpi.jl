export MPIROOT, isroot, mpi_shared_array, gather, bcast, allreduce

const MPIROOT = 0

struct ThreadComm end  # used for when thread only mode
const AbstractComm = Union{Comm,ThreadComm}

ismpi(comm::AbstractComm) = comm isa Comm

"""Assumes that a node comm has been passed"""
nnodes(comm::Comm) = MPI.Comm_size(comm)
nnodes(::ThreadComm) = 1

nprocs(comm::AbstractComm) = nnodes(comm) * Threads.nthreads()

noderank(comm::Comm) = MPI.Comm_rank(comm)
noderank(::ThreadComm) = 0

procrank(comm::AbstractComm) = (noderank(comm) * Threads.nthreads()) + (Threads.threadid() - 1)

isroot(comm::AbstractComm)::Bool = procrank(comm) == MPIROOT

# scatter(item, comm::Comm; root=MPIROOT) = MPI.Scatter(fill(item, nnodes(comm)), typeof(item), root, comm)
gather(items, comm::Comm; root = MPIROOT) = MPI.Gather(items, root, comm)
bcast(item, comm::Comm; root = MPIROOT) = MPI.bcast(item, root, comm)
allreduce(item, op, comm::Comm) = MPI.Allreduce(item, op, comm)

"""
Create a shared array, allocated by process with rank `owner_rank` on the
node_comm provided (i.e. when `MPI.Comm_rank(node_comm) == owner_rank`). Assumes all
processes on the node_comm are on the same node, or, more precisely that they
can create/access a shared mem block between them.
usage:
nrows, ncols = 100, 11
const arr = mpi_shared_array(MPI.COMM_WORLD, Int, (nrows, nworkers_node), owner_rank=0)

https://github.com/JuliaParallel/MPI.jl/blob/6f7f2336576408deb67b1e9080a6a9aa4144b067/test/test_shared_win.jl
"""
function mpi_shared_array(node_comm::MPI.Comm, ::Type{T}, sz::Tuple{Vararg{Int}}; owner_rank = 0) where {T}
    node_rank = MPI.Comm_rank(node_comm)
    len_to_alloc = MPI.Comm_rank(node_comm) == owner_rank ? prod(sz) : 0
    win, bufptr = MPI.Win_allocate_shared(T, len_to_alloc, node_comm)

    if node_rank != owner_rank
        len, sizofT, bufvoidptr = MPI.Win_shared_query(win, owner_rank)
        bufptr = convert(Ptr{T}, bufvoidptr)
    end
    win, unsafe_wrap(Array, bufptr, sz)
end