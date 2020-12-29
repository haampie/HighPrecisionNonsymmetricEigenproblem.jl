using VectorizationBase: Vec, MM, mask, pick_vector_width_shift, pick_vector_width, extractelement

using Base.Cartesian: @nexprs, @ntuple
using Base: IEEEFloat

import LinearAlgebra: lmul!, rmul!, dot, norm, axpy!

@generated function rmul!(A::StructArray{MultiFloat{T,N}, 2}, G::Rotation2{MultiFloat{T,N},MultiFloat{T,N}}) where {T<:IEEEFloat,N}

    W, Wshift = pick_vector_width_shift(T)

    quote
        @nexprs $N k -> p_k = stridedpointer(reinterpret(T, getfield(A, 1)[k]))

        m = size(A, 1)

        i = 1

        for step = Base.OneTo(m >> $Wshift)
            @nexprs $N k -> v_k_1 = vload(p_k, (MM{$W}(i), G.i))
            @nexprs $N k -> v_k_2 = vload(p_k, (MM{$W}(i), G.i+1))

            a₁ = MultiFloat(@ntuple $N k -> v_k_1)
            a₂ = MultiFloat(@ntuple $N k -> v_k_2)

            a₁′ = a₁ * Vec{$W,$T}(G.c) + a₂ * Vec{$W,$T}(G.s)
            a₂′ = a₁ * -Vec{$W,$T}(G.s) + a₂ * Vec{$W,$T}(G.c)

            @nexprs $N k -> vstore!(p_k, a₁′._limbs[k], (MM{$W}(i), G.i))
            @nexprs $N k -> vstore!(p_k, a₂′._limbs[k], (MM{$W}(i), G.i+1))

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && return A
        remainder_mask = mask($T, remainder)

        # Finally do the masked load, multiply and store.

        @nexprs $N k -> v_k_1 = vload(p_k, (MM{$W}(i), G.i), remainder_mask)
        @nexprs $N k -> v_k_2 = vload(p_k, (MM{$W}(i), G.i+1), remainder_mask)

        a₁ = MultiFloat(@ntuple $N k -> v_k_1)
        a₂ = MultiFloat(@ntuple $N k -> v_k_2)

        a₁′ = a₁ * Vec{$W,$T}(G.c) + a₂ * Vec{$W,$T}(G.s)
        a₂′ = a₁ * -Vec{$W,$T}(G.s) + a₂ * Vec{$W,$T}(G.c)

        @nexprs $N k -> vstore!(p_k, a₁′._limbs[k], (MM{$W}(i), G.i), remainder_mask)
        @nexprs $N k -> vstore!(p_k, a₂′._limbs[k], (MM{$W}(i), G.i+1), remainder_mask)

        A
    end
end

@generated function lmul!(G::Rotation2{MultiFloat{T,N},MultiFloat{T,N}}, A::StructArray{MultiFloat{T,N}, 2}) where {T<:IEEEFloat,N}

    W, Wshift = pick_vector_width_shift(T)

    quote
        @nexprs $N k -> p_k = stridedpointer(reinterpret(T, getfield(A, 1)[k]))

        m = size(A, 2)

        i = 1

        for step = Base.OneTo(m >> $Wshift)
            @nexprs $N k -> v_k_1 = vload(p_k, (G.i, MM{$W}(i)))
            @nexprs $N k -> v_k_2 = vload(p_k, (G.i+1, MM{$W}(i)))

            a₁ = MultiFloat(@ntuple $N k -> v_k_1)
            a₂ = MultiFloat(@ntuple $N k -> v_k_2)

            a₁′ = Vec{$W,$T}(G.c) * a₁ + Vec{$W,$T}(G.s) * a₂
            a₂′ = -Vec{$W,$T}(G.s) * a₁ + Vec{$W,$T}(G.c) * a₂

            @nexprs $N k -> vstore!(p_k, a₁′._limbs[k], (G.i, MM{$W}(i)))
            @nexprs $N k -> vstore!(p_k, a₂′._limbs[k], (G.i+1, MM{$W}(i)))

            i += $W
        end

        remainder = m & ($W - 1)
        remainder == 0 && return A
        remainder_mask = mask($T, remainder)

        # Finally do the masked load, multiply and store.
        @nexprs $N k -> v_k_1 = vload(p_k, (G.i, MM{$W}(i)), remainder_mask)
        @nexprs $N k -> v_k_2 = vload(p_k, (G.i+1, MM{$W}(i)), remainder_mask)

        a₁ = MultiFloat(@ntuple $N k -> v_k_1)
        a₂ = MultiFloat(@ntuple $N k -> v_k_2)

        a₁′ = Vec{$W,$T}(G.c) * a₁ + Vec{$W,$T}(G.s) * a₂
        a₂′ = -Vec{$W,$T}(G.s) * a₁ + Vec{$W,$T}(G.c) * a₂

        @nexprs $N k -> vstore!(p_k, a₁′._limbs[k], (G.i, MM{$W}(i)), remainder_mask)
        @nexprs $N k -> vstore!(p_k, a₂′._limbs[k], (G.i+1, MM{$W}(i)), remainder_mask)

        A
    end
end

@generated function MultiFloatOfVec(fs::NTuple{M,MultiFloat{T,N}}) where {T,M,N}
    exprs = [:(Vec($([:(fs[$j]._limbs[$i]) for j=1:M]...))) for i=1:N]

    return quote
        $(Expr(:meta, :inline))
        MultiFloat(tuple($(exprs...)))
    end
end

@generated function TupleOfMultiFloat(fs::MultiFloat{Vec{M,T},N}) where {T,M,N}
    exprs = [:(MultiFloat(tuple($([:(extractelement(fs._limbs[$j], $i)) for j=1:N]...)))) for i=0:M-1]
    return quote
        $(Expr(:meta, :inline))
        tuple($(exprs...))
    end
end

@generated function LinearAlgebra.dot(xs::AbstractArray{MultiFloat{T,N}}, ys::AbstractArray{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    load_xs = ntuple(k -> :(xs[i + $(k - 1)]), M)
    load_ys = ntuple(k -> :(ys[i + $(k - 1)]), M)

    quote
        vec_total = zero(MultiFloat{Vec{$M,$T},$N})

        # iterate in steps of M
        step, i = 1, 1
        @inbounds while step ≤ length(xs) ÷ $M
            x = MultiFloatOfVec(tuple($(load_xs...)))
            y = MultiFloatOfVec(tuple($(load_ys...)))
            vec_total += x * y
            step += 1
            i += $M
        end
    
        # sum the remainder
        @inbounds total = +(TupleOfMultiFloat(vec_total)...)
        while i ≤ length(xs)
            @inbounds total += xs[i] * ys[i]
            i += 1
        end

        return total
    end
end

@generated function LinearAlgebra.axpy!(a::MultiFloat{T,N}, xs::AbstractArray{MultiFloat{T,N}}, ys::AbstractArray{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    load_xs = ntuple(k -> :(xs[i + $(k - 1)]), M)
    load_ys = ntuple(k -> :(ys[i + $(k - 1)]), M)
    assign = [:($(load_ys[i]) = t[$i]) for i = 1:M]
    broadcast_a = ntuple(i -> :a, M)

    quote
        # iterate in steps of M
        step, i = 1, 1
        av = MultiFloatOfVec(tuple($(broadcast_a...)))
        @inbounds while step ≤ length(xs) ÷ $M
            x = MultiFloatOfVec(tuple($(load_xs...)))
            y = MultiFloatOfVec(tuple($(load_ys...)))
            result = y + x * av
            t = TupleOfMultiFloat(result)
            $(assign...)
            step += 1
            i += $M
        end
    
        # sum the remainder
        while i ≤ length(xs)
            @inbounds ys[i] += xs[i] * a
            i += 1
        end

        return ys
    end
end

LinearAlgebra.norm(xs::AbstractArray{<:MultiFloat}) = sqrt(dot(xs, xs))