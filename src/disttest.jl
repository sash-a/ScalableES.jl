using Distributed
p = addprocs(2, exeflags="--project")

# @everywhere using ParallelDataTransfer
# @everywhere include("policy.jl")
# @everywhere using .Plcy
# @everywhere using Flux


# function makepol()
#     global p
#     nn = Chain(Dense(5,5))
#     root_π = Plcy.Policy(nn)

#     @show root_π

#     sendto(p, π=root_π)

#     pmap(1:3) do i 
#         @show i
#         # @show π.θ
#         @show to_nn(π).layers[1].W
#         sleep(1)
#         π, 1, 4
#     end
# end


# x = map(fetch, makepol())
# for (p, o, f) in x
#     @show p o f
# end


# function stp(p)
#     sendto(p, pol=[1,2,3], obstat=4)

#     smthing = pmap(1:10) do i
#         @show i, pol
#         sleep(0.1)
#         pol
#     end

#     # futures = @sync @distributed for i in 1:10
#     #     @show i
#     #     pol + rand(1:10)
#     # end
#     @show smthing
#     smthing
# end


# fut = [fetch(f) for f in stp(p)]
# @show fut

@everywhere using SharedArrays

a = collect(1:100)
S = SharedArray{Int}((5), init = s->s[localindices(s)] = a[localindices(s)], pids=Int[1,2,3])
@everywhere struct T
    S
end
t = T(S)

function f(S)
    pmap(1:5) do i
        @show S[i]
        S[i] = -myid()
    end
end

