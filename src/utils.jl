import Random: rand

using Random, MultiFloats
using Random: AbstractRNG, SamplerType
using Base: IEEEFloat

Random.rand(rng::AbstractRNG, ::SamplerType{MultiFloat{T,N}}) where {T<:IEEEFloat,N} =
    renormalize(MultiFloat(ntuple(i -> rand(rng, T) * eps(T)^(i-1), Val{N}())))