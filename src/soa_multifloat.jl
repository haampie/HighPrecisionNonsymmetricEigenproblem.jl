using MultiFloats

import StructArrays
import Base: getproperty

# minimal effort to get MultiFloat struct of arrays to work.
# todo: complex.
Base.getproperty(a::MultiFloat, i::Int) = a._limbs[i]
StructArrays.staticschema(::Type{MultiFloat{T,M}}) where {T,M} = NTuple{M,T}
StructArrays.createinstance(::Type{MultiFloat{T,M}}, args::Vararg{T,M}) where {T,M} = MultiFloat{T,M}(values(args))

# better idea is to just define a 3d array.
struct MFArray{T,M,N,TA} <: AbstractArray{MultiFloat{T,M},N}
    A::TA
end

import Base: size, getindex, setindex, view, IndexStyle
using Base.Cartesian: @ntuple, @nexprs
export MFArray

Base.size(A::MFArray) = reverse(Base.tail(reverse(size(A.A))))

Base.IndexStyle(x::MFArray) = IndexStyle(x.A)

@generated function Base.getindex(A::MFArray{T,M,N}, i::Int) where {T,M,N}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        dist = length(A)
        MultiFloat{T,M}(@ntuple $M j -> A.A[i + (j - 1) * dist])
    end
end

@generated function Base.setindex!(A::MFArray{T,M,N}, v::MultiFloat{T,M}, i::Int) where {T,M,N}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        dist = length(A)
        @nexprs $M j -> A.A[i + (j-1) * dist] = v._limbs[j]
        return 1
    end
end

@generated function truncate_args(::Val{K}, x::Vararg{Any, N}) where {N,K}
    quote
        @ntuple $(min(N, K)) j -> x[j]
    end
end

@generated function Base.getindex(A::MFArray{T,M,N}, I::Vararg{Int,K}) where {T,M,N,K}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        ids = truncate_args(Val{N}(), I...)
        MultiFloat{T,M}(@ntuple $M j -> A.A[ids..., j])
    end
end

@generated function Base.setindex!(A::MFArray{T,M,N}, v::MultiFloat{T,M}, I::Vararg{Int,K}) where {T,M,N,K}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        ids = truncate_args(Val{N}(), I...)
        @nexprs $M j -> A.A[ids..., j] = v._limbs[j]
        return v
    end
end

function MFArray(A::AbstractArray{MultiFloat{T,M},N}) where {T,M,N}
    A′ = permutedims(reshape(reinterpret(T, A), M, size(A)...), circshift(OneTo(N+1), -1))
    return MFArray{T,M,N,typeof(A′)}(A′)
end

function view(A::MFArray{T,M,N}, I::Vararg{Any,K}) where {K,T,M,N}
    ids = truncate_args(Val{N}(), I...)
    B = view(A.A, ids..., :)
    return MFArray{T,M,ndims(B)-1,typeof(B)}(B)
end

using VectorizationBase: Unroll, VecUnroll

@generated function rmul!(A::MFArray, G::Rotation2{MultiFloat{T,M},MultiFloat{T,M}}) where {T<:Base.IEEEFloat,M}

    W, Wshift = pick_vector_width_shift(T)

    quote
        m = size(A.A, 1)
        i = 1
        pA = stridedpointer(A.A)

        for step = Base.OneTo(m >> $Wshift)

            I₁ = Unroll{3,1,$M,1,$W,zero(UInt)}((i, G.i  , 1))
            I₂ = Unroll{3,1,$M,1,$W,zero(UInt)}((i, G.i + 1, 1))
            
            a₁ = MultiFloat(vload(pA, I₁).data)
            a₂ = MultiFloat(vload(pA, I₂).data)

            a₁′ = a₁ * Vec{$W,$T}(G.c) + a₂ * Vec{$W,$T}(G.s)
            a₂′ = a₁ * -Vec{$W,$T}(G.s) + a₂ * Vec{$W,$T}(G.c)

            vstore!(pA, VecUnroll(a₁′._limbs), I₁)
            vstore!(pA, VecUnroll(a₂′._limbs), I₂)

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && return A
        remainder_mask = mask($T, remainder)

        I₁ = Unroll{3,1,$M,1,$W,typemax(UInt)}((i, G.i  , 1))
        I₂ = Unroll{3,1,$M,1,$W,typemax(UInt)}((i, G.i + 1, 1))
        
        a₁ = MultiFloat(vload(pA, I₁, remainder_mask).data)
        a₂ = MultiFloat(vload(pA, I₂, remainder_mask).data)

        a₁′ = a₁ * Vec{$W,$T}(G.c) + a₂ * Vec{$W,$T}(G.s)
        a₂′ = a₁ * -Vec{$W,$T}(G.s) + a₂ * Vec{$W,$T}(G.c)

        vstore!(pA, VecUnroll(a₁′._limbs), I₁, remainder_mask)
        vstore!(pA, VecUnroll(a₂′._limbs), I₂, remainder_mask)

        A
    end
end

@generated function lmul!(G::Rotation2{MultiFloat{T,M},MultiFloat{T,M}}, A::MFArray) where {T<:Base.IEEEFloat,M}

    W, Wshift = pick_vector_width_shift(T)

    quote
        m = size(A.A, 2)
        i = 1
        pA = stridedpointer(A.A)

        for step = Base.OneTo(m >> $Wshift)

            I₁ = Unroll{3,1,$M,2,$W,zero(UInt)}((G.i,     i, 1))
            I₂ = Unroll{3,1,$M,2,$W,zero(UInt)}((G.i + 1, i, 1))
            
            a₁ = MultiFloat(vload(pA, I₁).data)
            a₂ = MultiFloat(vload(pA, I₂).data)

            a₁′ = Vec{$W,$T}(G.c) * a₁ + Vec{$W,$T}(G.s) * a₂
            a₂′ = -Vec{$W,$T}(G.s) * a₁ + Vec{$W,$T}(G.c) * a₂

            vstore!(pA, VecUnroll(a₁′._limbs), I₁)
            vstore!(pA, VecUnroll(a₂′._limbs), I₂)

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && return A
        remainder_mask = mask($T, remainder)

        I₁ = Unroll{3,1,$M,2,$W,typemax(UInt)}((G.i    , i, 1))
        I₂ = Unroll{3,1,$M,2,$W,typemax(UInt)}((G.i + 1, i, 1))
        
        a₁ = MultiFloat(vload(pA, I₁, remainder_mask).data)
        a₂ = MultiFloat(vload(pA, I₂, remainder_mask).data)

        a₁′ = Vec{$W,$T}(G.c) * a₁ + Vec{$W,$T}(G.s) * a₂
        a₂′ = -Vec{$W,$T}(G.s) * a₁ + Vec{$W,$T}(G.c) * a₂

        vstore!(pA, VecUnroll(a₁′._limbs), I₁, remainder_mask)
        vstore!(pA, VecUnroll(a₂′._limbs), I₂, remainder_mask)

        A
    end
end

const MFVector{T,M} = MFArray{T,M,1}

using Base.Cartesian: @ntuple

import LinearAlgebra: axpy!

@generated function LinearAlgebra.axpy!(a::MultiFloat{T,M}, xs::MFVector{T,M}, ys::MFVector{T,M}) where {T,M}

    W, Wshift = pick_vector_width_shift(T)

    quote
        m = size(xs.A, 1)
        i = 1

        px = stridedpointer(xs.A)
        py = stridedpointer(ys.A)

        av = MultiFloat(@ntuple $M j -> Vec{$W,$T}(a._limbs[j]))

        for step = Base.OneTo(m >> $Wshift)
            I = Unroll{2,1,$M,1,$W,zero(UInt)}((i, 1))
            
            xi = MultiFloat(vload(px, I).data)
            yi = MultiFloat(vload(py, I).data)

            yi′ = xi * av + yi

            vstore!(py, VecUnroll(yi′._limbs), I)

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && return ys
        remainder_mask = mask($T, remainder)

        I = Unroll{2,1,$M,1,$W,typemax(UInt)}((i, 1))
            
        xi = MultiFloat(vload(px, I, remainder_mask).data)
        yi = MultiFloat(vload(py, I, remainder_mask).data)

        yi′ = xi * av + yi

        vstore!(py, VecUnroll(yi′._limbs), I, remainder_mask)

        return ys
    end
end