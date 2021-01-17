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

const Q_WIDTH = 4
const Q_HEIGHT = 32

function A_panel_width(::Type{MultiFloat{T, M}}) where {T, M}
    W = pick_vector_width(T)
    return W * (((VectorizationBase.L₁CACHE.size ÷ sizeof(T) ÷ M - Q_WIDTH) ÷ Q_HEIGHT - Q_WIDTH) ÷ W)
end

function to_buffer(Q::AbstractMatrix{MultiFloat{Float64,M}}, τs::AbstractVector{MultiFloat{Float64,M}}, A::AbstractMatrix{MultiFloat{Float64,M}}) where {M}
    m, n = size(A)

    MF_T = MultiFloat{Float64, M}
    W = pick_vector_width(Float64)

    @assert size(Q, 1) == m == Q_HEIGHT
    @assert size(Q, 2) == length(τs) == Q_WIDTH
    @assert n % A_panel_width(MF_T) == 0

    A_size = M * Q_HEIGHT * n
    Q_size = M * Q_HEIGHT * Q_WIDTH
    τ_size = M * Q_WIDTH

    a = max(VectorizationBase.REGISTER_SIZE, VectorizationBase.L₁CACHE.linesize)
    resize!(BUFFER, (A_size + Q_size + τ_size) * sizeof(Float64) + 3a)

    p = VectorizationBase.align(convert(Ptr{Float64}, pointer(BUFFER)), a)
    τs′ = unsafe_wrap(Array, p, τ_size, own=false)
    p = VectorizationBase.align(p + sizeof(Float64) * τ_size, a)
    Q′ = unsafe_wrap(Array, p, Q_size, own=false)
    p = VectorizationBase.align(p + sizeof(Float64) * Q_size, a)
    A′ = unsafe_wrap(Array, p, A_size, own=false)

    # Copy τ, Q and A into the buffers.
    # not gonna order 
    copyto!(τs′,  reinterpret(Float64, τs))

    # Q is simply column major.
    copyto!(Q′, reinterpret(Float64, Q))

    # And for A we access
    copyto!(A′, permutedims(reshape(reinterpret(Float64, A), M * Q_HEIGHT, W, n ÷ W), (2, 1, 3)))

    return Q′, τs′, A′
end

function unpack!(A::AbstractMatrix{MultiFloat{Float64,M}}, Abuf::Vector{Float64}) where {M}
    m, n = size(A)

    W = pick_vector_width(Float64)

    @assert size(A, 1) == Q_HEIGHT
    @assert length(A) == length(Abuf) ÷ M

    # And for A we acces
    copyto!(A, reinterpret(MultiFloat{Float64,M}, permutedims(reshape(Abuf, W, M * Q_HEIGHT, n ÷ W), (2, 1, 3))))

    return A
end

using Base.Cartesian: @nexprs, @ntuple

@generated function do_qr!(packedQ, packedτ, packedA, ::Type{MultiFloat{Float64,M}}) where {M}
    # minikernel for size.
    # τ is stored in buffer[1:4] (when interpreted as Float64xM)
    # Q is stored in buffer[5:end][:, 1:4] (interpreted as Matrix{Float64xM} of size 34xN)
    # A is stored in buffer[5:end][:, 5:end] (interpreted as Matrix{Float64xM} of size 34xN)

    A_width = A_panel_width(MultiFloat{Float64, M})
    W = pick_vector_width(Float64)

    quote
        pA = stridedpointer(packedA)
        panel = 0
        @inbounds while panel < length(packedA)
            τi = 1
            for q_col = OneTo(4)
                # broadcast τ's limbs to full-width registers.
                @nexprs $M j -> τ_j = Vec{$W}(packedτ[τi + j - 1])

                τ_mf = MultiFloat(@ntuple $M j -> τ_j)
                
                # Reduce columns [col_offset, col_offset+4)
                aj = panel + $M * $W * (q_col - 1) + 1

                # Loop over the columns in block of size W
                for col_block = OneTo($A_width ÷ $W)

                    qi = (q_col - 1) * $M * Q_HEIGHT + $M * (q_col - 1) + 1

                    start_ai = aj
                    start_qi = qi

                    # Accumulator. reflector[1] == 1, so no multiplication here.
                    @nexprs $M j -> a_j = vload(pA, (MM{$W}(aj + $W * (j - 1)),))

                    accumulator_mf = MultiFloat(@ntuple $M j -> a_j)

                    aj += $M * $W
                    qi += $M

                    # dot over the next 31 rows.
                    for row = OneTo(Q_HEIGHT - q_col)
                        # Load the Q-values and broadcast them to a single register
                        @nexprs $M j -> q_j = Vec{$W}(packedQ[qi + j - 1])
                        q_mf = MultiFloat(@ntuple $M j -> q_j)

                        # Load the values of A of 4 columns
                        @nexprs $M j -> a_j = vload(pA, (MM{$W}(aj + $W * (j - 1)),))
                        a_mf = MultiFloat(@ntuple $M j -> a_j)
                        
                        accumulator_mf += q_mf * a_mf

                        qi += $M
                        aj += $W * $M
                    end

                    α_mf = τ_mf * accumulator_mf

                    # go back to the top.
                    aj = start_ai
                    qi = start_qi

                    @nexprs $M j -> a_j = vload(pA, (MM{$W}(aj + $W * (j - 1)),))

                    a_mf = MultiFloat(@ntuple $M j -> a_j)

                    a_mf -= α_mf

                    @nexprs $M j -> vstore!(pA, a_mf._limbs[j], (MM{$W}(aj + $W * (j - 1)),))

                    aj += $M * $W
                    qi += $M

                    # Finally we do the axpy a -= α * q
                    for row = OneTo(Q_HEIGHT - q_col)
                        # Load the Q-values and broadcast them to a single register
                        @nexprs $M j -> q_j = Vec{$W}(packedQ[qi + j - 1])

                        q_mf = MultiFloat(@ntuple $M j -> q_j)

                        # Load the values of A of 4 columns
                        @nexprs $M j -> a_j = vload(pA, (MM{$W}(aj + $W * (j - 1)),))
                        a_mf = MultiFloat(@ntuple $M j -> a_j)

                        a_mf -= α_mf * q_mf

                        @nexprs $M j -> vstore!(pA, a_mf._limbs[j], (MM{$W}(aj + $W * (j - 1)),))

                        aj += $M * $W
                        qi += $M
                    end

                    aj += $M * $W * (q_col - 1)
                end

                τi += $M
            end

            panel += Q_HEIGHT * $A_width * $M
        end
    end
end