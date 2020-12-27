module HighPrecisionNonsymmetricEigenproblem

using StructArrays, VectorizationBase, LinearAlgebra


"""
Passed into the Schur factorization function if you do not wish to have the Schur vectors.
"""
struct NotWanted end

abstract type SmallRotation end

"""
Given's rotation acting on rows i:i+1
"""
struct Rotation2{Tc,Ts} <: SmallRotation
    c::Tc
    s::Ts
    i::Int
end

"""
Two Given's rotations acting on rows i:i+2. This could also be implemented as one Householder
reflector!
"""
struct Rotation3{Tc,Ts} <: SmallRotation
    c₁::Tc
    s₁::Ts
    c₂::Tc
    s₂::Ts
    i::Int
end


include("tsqr.jl")
include("hessenberg.jl")
include("givens.jl")
include("soa_multifloat.jl")
include("schur.jl")
include("utils.jl")
include("multiprecision.jl")

end
