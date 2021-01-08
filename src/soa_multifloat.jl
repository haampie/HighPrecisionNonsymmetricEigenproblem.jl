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
        return v
    end
end

@generated function Base.getindex(A::MFArray{T,M,N}, I::Vararg{Int,N}) where {T,M,N}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        MultiFloat{T,M}(@ntuple $M j -> A.A[I..., j])
    end
end

@generated function Base.setindex!(A::MFArray{T,M,N}, v::MultiFloat{T,M}, I::Vararg{Int,N}) where {T,M,N}
    quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        @nexprs $M j -> A.A[I..., j] = v._limbs[j]
        return v
    end
end

function MFArray(A::AbstractArray{MultiFloat{T,M},N}) where {T,M,N}
    A′ = permutedims(reshape(reinterpret(T, A), M, size(A)...), circshift(OneTo(N+1), -1))
    return MFArray{T,M,N,typeof(A′)}(A′)
end

@inline function view(A::MFArray{T,M,N}, I::Vararg{Any,N}) where {T,M,N}
    B = view(A.A, I..., :)
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

import LinearAlgebra: axpy!, dot

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

@generated function LinearAlgebra.dot(xs::MFVector{T,M}, ys::MFVector{T,M}) where {T,M}
    W, Wshift = pick_vector_width_shift(T)

    quote
        m = size(xs.A, 1)
        i = 1

        px = stridedpointer(xs.A)
        py = stridedpointer(ys.A)

        vec_total = zero(MultiFloat{Vec{$W,$T},$M})

        for step = Base.OneTo(m >> $Wshift)
            I = Unroll{2,1,$M,1,$W,zero(UInt)}((i, 1))
            
            xi = MultiFloat(vload(px, I).data)
            yi = MultiFloat(vload(py, I).data)

            vec_total += xi * yi

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && @goto sum_scalar
        remainder_mask = mask($T, remainder)

        I = Unroll{2,1,$M,1,$W,typemax(UInt)}((i, 1))
            
        xi = MultiFloat(vload(px, I, remainder_mask).data)
        yi = MultiFloat(vload(py, I, remainder_mask).data)

        vec_total += xi * yi

        @label sum_scalar
        return +(TupleOfMultiFloat(vec_total)...)
    end
end

const BUFFER = Vector{UInt8}(undef, 0)

function to_buffer(Q::AbstractMatrix{Float64x4}, τs::AbstractVector{Float64x4}, A::AbstractMatrix{Float64x4})
    m, n = size(A)

    @assert size(Q, 1) == m == 32
    @assert size(Q, 2) == length(τs) == 4
    @assert n % 28 == 0

    A_size = 4 * 32 * n
    Q_size = 4 * 32 * 4
    τ_size = 4

    a = max(VectorizationBase.REGISTER_SIZE, VectorizationBase.L₁CACHE.linesize)
    resize!(BUFFER, (A_size + Q_size + τ_size) * sizeof(Float64) + 3a)

    p = VectorizationBase.align(convert(Ptr{Float64}, pointer(BUFFER)), a)
    τs′ = unsafe_wrap(Array, p, (4, 4), own=false)
    p = VectorizationBase.align(p + sizeof(Float64) * length(τs′), a)
    Q′ = unsafe_wrap(Array, p, 4 * 32 * 4, own=false)
    p = VectorizationBase.align(p + sizeof(Float64) * length(Q′), a)
    A′ = unsafe_wrap(Array, p, 4 * 32 * n, own=false)

    # Copy τ, Q and A into the buffers.
    # not gonna order 
    copyto!(τs′,  reinterpret(Float64, τs))

    # Q is simply column major.
    copyto!(Q′, reinterpret(Float64, Q))

    # And for A we access
    copyto!(A′, permutedims(reshape(permutedims(reshape(reinterpret(Float64, A), 4, 32, n), (1, 3, 2)), 4, 4, n ÷ 4, 32), (2, 1, 4, 3)))

    return Q′, τs′, A′
end

function unpack!(A::AbstractMatrix{Float64x4}, Abuf::Vector{Float64})
    m, n = size(A)

    @assert size(A, 1) == 32
    @assert length(A) == length(Abuf) ÷ 4

    # And for A we acces
    copyto!(A, reinterpret(Float64x4, reshape(permutedims(reshape(permutedims(reshape(Abuf, 4, 4, 32, n ÷ 4), (2, 1, 4, 3)), 4, n, 32), (1, 3, 2)), 128, n)))

    return A
end

function do_qr!(packedQ, packedτ, packedA)
    # minikernel for size.
    # τ is stored in buffer[1:4] (when interpreted as Float64x4)
    # Q is stored in buffer[5:end][:, 1:4] (interpreted as Matrix{Float64x4} of size 34xN)
    # A is stored in buffer[5:end][:, 5:end] (interpreted as Matrix{Float64x4} of size 34xN)

    pA = stridedpointer(packedA)
    # pQ = stridedpointer(packedQ)
    # pτ = stridedpointer(packedτ)

    @inbounds for panel = OneTo(length(packedA) ÷ (32 * 28 * 4))
        τi = 1
        for q_col = OneTo(4)
            # broadcast τ's limbs to full-width registers.
            τ1 = Vec{4}(packedτ[τi+0])
            τ2 = Vec{4}(packedτ[τi+1])
            τ3 = Vec{4}(packedτ[τi+2])
            τ4 = Vec{4}(packedτ[τi+3])

            τ_mf = MultiFloat((τ1, τ2, τ3, τ4))
            
            # Reduce columns [col_offset, col_offset+4)
            aj = 32 * 28 * 4 * (panel - 1) + 16 * (q_col - 1) + 1

            # Loop over the columns in block of size 4.
            for col_block = OneTo(28 ÷ 4)

                qi = (q_col - 1) * 4 * 32 + 4 * (q_col - 1) + 1

                start_ai = aj
                start_qi = qi

                # Accumulator. reflector[1] == 1, so no multiplication here.
                a11 = vload(pA, (MM{4}(aj + 0 ),))
                a21 = vload(pA, (MM{4}(aj + 4 ),))
                a31 = vload(pA, (MM{4}(aj + 8 ),))
                a41 = vload(pA, (MM{4}(aj + 12),))

                accumulator_mf = MultiFloat((a11, a21, a31, a41))

                aj += 16
                qi += 4

                # dot over the next 31 rows.
                for row = OneTo(32 - q_col)
                    # Load the Q-values and broadcast them to a single register
                    q1 = Vec{4}(packedQ[qi + 0])
                    q2 = Vec{4}(packedQ[qi + 1])
                    q3 = Vec{4}(packedQ[qi + 2])
                    q4 = Vec{4}(packedQ[qi + 3])

                    q_mf = MultiFloat((q1, q2, q3, q4))

                    # Load the values of A of 4 columns
                    a1 = vload(pA, (MM{4}(aj + 0),))
                    a2 = vload(pA, (MM{4}(aj + 4),))
                    a3 = vload(pA, (MM{4}(aj + 8),))
                    a4 = vload(pA, (MM{4}(aj + 12),))

                    a_mf = MultiFloat((a1, a2, a3, a4))
                    
                    accumulator_mf += q_mf * a_mf

                    qi += 4
                    aj += 16
                end

                α_mf = τ_mf * accumulator_mf

                # go back to the top.
                aj = start_ai
                qi = start_qi

                a1 = vload(pA, (MM{4}(aj + 0),))
                a2 = vload(pA, (MM{4}(aj + 4),))
                a3 = vload(pA, (MM{4}(aj + 8),))
                a4 = vload(pA, (MM{4}(aj + 12),))

                a_mf = MultiFloat((a1, a2, a3, a4))

                a_mf -= α_mf

                vstore!(pA, a_mf._limbs[1], (MM{4}(aj + 0 ),))
                vstore!(pA, a_mf._limbs[2], (MM{4}(aj + 4 ),))
                vstore!(pA, a_mf._limbs[3], (MM{4}(aj + 8 ),))
                vstore!(pA, a_mf._limbs[4], (MM{4}(aj + 12),))

                aj += 16
                qi += 4

                # Finally we do the axpy a -= α * q
                for row = OneTo(32 - q_col)
                    # Load the Q-values and broadcast them to a single register
                    q1 = Vec{4}(packedQ[qi+0])
                    q2 = Vec{4}(packedQ[qi+1])
                    q3 = Vec{4}(packedQ[qi+2])
                    q4 = Vec{4}(packedQ[qi+3])

                    q_mf = MultiFloat((q1, q2, q3, q4))

                    # Load the values of A of 4 columns
                    a1 = vload(pA, (MM{4}(aj + 0),))
                    a2 = vload(pA, (MM{4}(aj + 4),))
                    a3 = vload(pA, (MM{4}(aj + 8),))
                    a4 = vload(pA, (MM{4}(aj + 12),))

                    a_mf = MultiFloat((a1, a2, a3, a4))

                    a_mf -= α_mf * q_mf

                    vstore!(pA, a_mf._limbs[1], (MM{4}(aj + 0),))
                    vstore!(pA, a_mf._limbs[2], (MM{4}(aj + 4),))
                    vstore!(pA, a_mf._limbs[3], (MM{4}(aj + 8),))
                    vstore!(pA, a_mf._limbs[4], (MM{4}(aj + 12),))

                    qi += 4
                    aj += 16
                end

                aj += 16 * (q_col - 1)
            end

            τi += 4
        end
    end
end