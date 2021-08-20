module MpiUtils

import MPI: MPI, Comm
import Base: size


export MPIROOT, size, isroot, mpi_shared_array, gather, bcast, allreduce

MPIROOT = 0

isroot(comm::Comm)::Bool = MPI.Comm_rank(comm) == MPIROOT
size(comm::Comm) = MPI.Comm_size(comm)
gather(items, comm::Comm; root=MPIROOT) = MPI.Gather(items, root, comm)
# scatter(item, comm::Comm; root=MPIROOT) = MPI.Scatter(fill(item, size(comm)), typeof(item), root, comm)
bcast(item, comm::Comm; root=MPIROOT) = MPI.bcast(item, root, comm)
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
function mpi_shared_array(node_comm::MPI.Comm, ::Type{T}, sz::Tuple{Vararg{Int}}; owner_rank=0) where T
    node_rank = MPI.Comm_rank(node_comm)
    len_to_alloc = MPI.Comm_rank(node_comm) == owner_rank ? prod(sz) : 0
    win, bufptr = MPI.Win_allocate_shared(T, len_to_alloc, node_comm)

    if node_rank != owner_rank
        len, sizofT, bufvoidptr = MPI.Win_shared_query(win, owner_rank)
        bufptr = convert(Ptr{T}, bufvoidptr)
    end
    win, unsafe_wrap(Array, bufptr, sz)
end

end