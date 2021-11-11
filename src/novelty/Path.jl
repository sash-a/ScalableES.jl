struct Vec2{T}
    x::T
    y::T
end

# typedefs
const Path = Vector{Vec2{Float64}}
const SPath{S} = SVector{S,Vec2{Float64}}
const AbstractPath = Union{Path,SPath}

Base.vcat(p::AbstractPath) = vcat(map(v -> [v.x, v.y], p)...)
Distances.euclidean(x::Path, y::Path) = Distances.euclidean(vcat(x), vcat(y))

to_path(a::AbstractVector{T}) where T <: Number  = Path([Vec2{T}(x, y) for (x, y) in collect(partition(a, 2))])