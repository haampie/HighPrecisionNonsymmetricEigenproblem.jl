using VectorizationBase: Vec, MM, mask, pick_vector_width_shift

using Base.Cartesian: @nexprs, @ntuple

@generated function rmul!(A::StructArray{MultiFloat{T,N}, 2}, G::Rotation2{MultiFloat{T,N},MultiFloat{T,N}}) where {T,N}

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

@generated function lmul!(G::Rotation2{MultiFloat{T,N},MultiFloat{T,N}}, A::StructArray{MultiFloat{T,N}, 2}) where {T,N}

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