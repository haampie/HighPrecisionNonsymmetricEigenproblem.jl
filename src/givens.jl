import LinearAlgebra: lmul!, rmul!
import Base: Matrix

function compute_rotate(f, g)
    r = hypot(f, g)
    cs = f / r
    sn = g / r
    if cs < 0 && abs(f) > abs(g)
        cs = -cs
        sn = -sn
        r = -r
    end
    return cs, sn, r
end

# Some utility to materialize a rotation to a matrix.
function Matrix(r::Rotation2{Tc,Ts}, n::Int) where {Tc,Ts}
    r.i < n || throw(ArgumentError("Matrix should have order $(r.i+1) or larger"))
    G = Matrix{promote_type(Tc,Ts)}(I, n, n)
    G[r.i+0,r.i+0] = r.c
    G[r.i+1,r.i+0] = -conj(r.s)
    G[r.i+0,r.i+1] = r.s
    G[r.i+1,r.i+1] = r.c
    return G
end

function Matrix(r::Rotation3{Tc,Ts}, n::Int) where {Tc,Ts}
    G₁ = Matrix(Rotation2(r.c₁, r.s₁, r.i + 1), n)
    G₂ = Matrix(Rotation2(r.c₂, r.s₂, r.i), n)
    return G₂ * G₁
end

"""
Get a rotation that maps [p₁, p₂] to a multiple of [1, 0]
"""
function get_rotation(p₁, p₂, i::Int)
    c, s, nrm = compute_rotate(p₁, p₂)
    Rotation2(c, s, i), nrm
end

"""
Get a rotation that maps [p₁, p₂, p₃] to a multiple of [1, 0, 0]
"""
function get_rotation(p₁, p₂, p₃, i::Int)
    c₁, s₁, nrm₁ = compute_rotate(p₂, p₃)
    c₂, s₂, nrm₂ = compute_rotate(p₁, nrm₁)
    Rotation3(c₁, s₁, c₂, s₂, i), nrm₂
end

lmul!(::SmallRotation, ::NotWanted, args...) = nothing
rmul!(::NotWanted, ::SmallRotation, args...) = nothing

@inline function lmul!(G::Rotation3, A::AbstractMatrix)
    @inbounds for j = axes(A, 2)
        a₁ = A[G.i+0,j]
        a₂ = A[G.i+1,j]
        a₃ = A[G.i+2,j]

        a₂′ = G.c₁ * a₂ + G.s₁ * a₃
        a₃′ = -G.s₁' * a₂ + G.c₁ * a₃

        a₁′′ = G.c₂ * a₁ + G.s₂ * a₂′
        a₂′′ = -G.s₂' * a₁ + G.c₂ * a₂′
        
        A[G.i+0,j] = a₁′′
        A[G.i+1,j] = a₂′′
        A[G.i+2,j] = a₃′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation3)
    @inbounds for j = axes(A, 1)
        a₁ = A[j,G.i+0]
        a₂ = A[j,G.i+1]
        a₃ = A[j,G.i+2]

        a₂′ = a₂ * G.c₁ + a₃ * G.s₁'
        a₃′ = a₂ * -G.s₁ + a₃ * G.c₁

        a₁′′ = a₁ * G.c₂ + a₂′ * G.s₂'
        a₂′′ = a₁ * -G.s₂ + a₂′ * G.c₂

        A[j,G.i+0] = a₁′′
        A[j,G.i+1] = a₂′′
        A[j,G.i+2] = a₃′
    end
    A
end

@inline function lmul!(G::Rotation2, A::AbstractMatrix)
    @inbounds for j = axes(A, 2)
        a₁ = A[G.i+0,j]
        a₂ = A[G.i+1,j]

        a₁′ = G.c * a₁ + G.s * a₂
        a₂′ = -G.s' * a₁ + G.c * a₂

        A[G.i+0,j] = a₁′
        A[G.i+1,j] = a₂′
    end

    A
end

@inline function rmul!(A::AbstractMatrix, G::Rotation2)
    @inbounds for j = axes(A, 1)
        a₁ = A[j,G.i+0]
        a₂ = A[j,G.i+1]

        a₁′ = a₁ * G.c + a₂ * G.s'
        a₂′ = a₁ * -G.s + a₂ * G.c

        A[j,G.i+0] = a₁′
        A[j,G.i+1] = a₂′
    end
    A
end