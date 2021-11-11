struct ArchiveEntry{T}
    novelty::T
    behaviour::Path
end

Archive = Vector{ArchiveEntry{Float64}}

"""
Run policies through `behv_fn` and obtain behaviour vector then compute novelty between all behaviours

behv_fn takes in a policy and returns it's behaviour
"""
function init_archive(policies::Vector{T}, behv_fn) where T <: AbstractPolicy
    behaviours = Vector{Path}()
    for p in policies
        batched_behvs = behv_fn(to_nn(p))
        push!(behaviours, mean_bc(batched_behvs))   
    end

    # calculate novelty of each behaviour to all other behaviours excluding self
    novelties = map(b -> novelty(b, filter(cmp -> b != cmp, behaviours), 10), behaviours)
    [ArchiveEntry(n, b) for (n, b) in zip(novelties, behaviours)]
end


function update_archive!(archive::Archive, p::AbstractPolicy, k::Int, behv_fn)
    behaviour = mean_bc(behv_fn(to_nn(p)))
    push!(archive, ArchiveEntry(novelty(behaviour, archive, k), behaviour))
end

# TODO does mean path make sense?
function mean_bc(behaviours::AbstractVector{T}) where T <: AbstractPath
    a = [vcat(b) for b in behaviours]
    to_path(mean(a))
end


"""Different forms of computing the novelty of a behaviour against the archive"""
novelty(behaviours, archive::Archive, n::Int) = novelty(behaviours, map(a->a.behaviour, archive), n)
novelty(result::NsEsResult, archive::Archive, n::Int) = novelty(result.behaviours, map(a->a.behaviour, archive), n)

function novelty(batch_behaviours::AbstractVector{T}, archive_behaviours::AbstractVector{U}, n::Int) where T <: AbstractPath where U <: AbstractPath
    # map(b -> novelty(b, archive_behaviours, n), batch_behaviours) / length(batch_behaviours)
    novelty(mean_bc(batch_behaviours), archive_behaviours, n)
end

function novelty(behaviour::AbstractPath, archive_behaviours::AbstractVector{T}, n::Int) where T <: AbstractPath
    h = MutableBinaryMinHeap{Float64}()
    for b in archive_behaviours  
        push!(h, euclidean(behaviour, b))  # get distance between two paths and add to heap
    end

    n = min(n, length(archive_behaviours))
    sumsmallest = sum([pop!(h) for _ in 1:n])
    sumsmallest / n  # get nsmallest and return mean
end

