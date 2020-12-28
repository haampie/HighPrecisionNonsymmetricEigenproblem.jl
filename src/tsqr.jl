using Base: OneTo

using Base.Threads: @spawn
using Base: @sync

struct Panel
    w::Int
    h::Int
end

raw"""
Computes the amount of τs we have to store for all
reflectors (I - τvv') in a tsqr. If we have 2 stacked panels
of width w, we need 3 small qr's using w reflectors each, so
3w τs.
x   x
 \ /
  x

Another example is 5 stacked panels, will go like this:
x   x   x   x   x
 \ /     \ /   /
  x       x   /
   \     /   /
    \   /   /
     \ /   /
      x   /
       \ /
        x
It's effectively 9 small qr's with w reflectors, so 9w τs.
Generaly the formula is simply panel_width * (2 * num_panels_vertically - 1)
"""
function reflectors_per_tsqr(matrix_height, p::Panel)
    num_panels_vertically = ÷(matrix_height, p.h, RoundUp)
    reflectors_per_panel = p.w #assumes p.h ≥ p.w
    return reflectors_per_panel * (2 * num_panels_vertically - 1)
end

function to_band!(A::AbstractMatrix{T}, p = Panel(16, 32)) where {T}
    @assert size(A, 1) == size(A, 2)
    @assert p.w ≤ p.h

    # on top we have a section of size p.w x p.w
    m = size(A, 1)

    τs = Vector{T}(undef, reflectors_per_tsqr(m - p.w, p))

    # loop over all column blocks
    for b = OneTo((m - p.w) ÷ p.w)

        # divide into the bit we're QR'ing
        # the bit to which we apply it from the left
        # and the bit to which we apply it from the right.
        curr_block = (b - 1) * p.w + 1
        next_block = curr_block + p.w

        A_qr = view(A, next_block:m, curr_block:next_block-1)

        tsqr!(A_qr, τs, p)

        # Now apply the tsqr bit from the left and the right to A.
        A_left = view(A, next_block:m, next_block:m)
        A_right = view(A, :, next_block:m)

        apply_tsqr_left!(A_qr, A_left, τs, p)
        apply_tsqr_right!(A_qr, A_right, τs, p)
    end

    return A
end

function exponent_of_2_greater_than_or_equal_to(m)
    k = 0
    two_to_the_k = 1
    while two_to_the_k < m
        two_to_the_k *= 2
        k += 1
    end

    return k
end

function tsqr!(A, τs, p::Panel)
    m, n = size(A)

    @assert n == p.w

    num_panels_vertically = ÷(m, p.h, RoundUp)

    # Keep the τ offset conveniently here, cause
    # we need that when walking up the tree later.
    τ_offset = 1

    # First standard QR on every panel.
    for b = OneTo(num_panels_vertically)
        b_start = (b - 1) * p.h + 1
        b_end = min(b_start + p.h - 1, m)
        τs′ = view(τs, τ_offset:τ_offset+p.w-1)
        A′ = view(A, b_start:b_end, :)

        block_qr!(A′, τs′)

        τ_offset += p.w
    end

    # Then we walk up the tree and do QR on upper triangular blocks.
    tree_height = exponent_of_2_greater_than_or_equal_to(num_panels_vertically)

    # we're combining two panels of height `merged_block_size`
    merged_block_size = p.h

    # we've already done the leaf nodes, so - 1.
    for level = OneTo(tree_height)
        num_panels_vertically = ÷(m, merged_block_size, RoundUp)

        # combine all pairs of panels.
        for b = OneTo(num_panels_vertically ÷ 2)
            b_start = (b - 1) * 2merged_block_size + 1
            b_end = min(b_start + 2merged_block_size - 1, m)

            A′ = view(A, b_start:b_end, :)
            τs′ = view(τs, τ_offset:τ_offset+p.w-1)
            
            #=@spawn=# merge_block_qr!(A′, τs′, merged_block_size)
            
            τ_offset += p.w
        end

        merged_block_size *= 2
    end
end

function apply_tsqr_left!(qr, A, τs, p::Panel)
    m, n = size(qr)

    @assert n == p.w

    num_panels_vertically = ÷(m, p.h, RoundUp)

    # Keep the τ offset conveniently here, cause
    # we need that when walking up the tree flater.
    τ_offset = 1

    # First standard QR on every panel.
    for b = OneTo(num_panels_vertically)
        b_start = (b - 1) * p.h + 1
        b_end = min(b_start + p.h - 1, m)
        τs′ = view(τs, τ_offset:τ_offset+p.w-1)

        qr′ = view(qr, b_start:b_end, :)
        A′ = view(A, b_start:b_end, :)

        #=@spawn=# apply_tiny_Q_left!(qr′, A′, τs′)

        τ_offset += p.w
    end

    # Then we walk up the tree and do QR on upper triangular blocks.
    tree_height = exponent_of_2_greater_than_or_equal_to(num_panels_vertically)

    # we're combining two panels of height `merged_block_size`
    merged_block_size = p.h

    # we've already done the leaf nodes, so - 1.
    for level = OneTo(tree_height)
        num_panels_vertically = ÷(m, merged_block_size, RoundUp)
        
        # combine all pairs of panels.
        for b = OneTo(num_panels_vertically ÷ 2)
            b_start = (b - 1) * 2merged_block_size + 1
            b_end = min(b_start + 2merged_block_size - 1, m)

            qr′ = view(qr, b_start:b_end, :)
            A′ = view(A, b_start:b_end, :)
            τs′ = view(τs, τ_offset:τ_offset+p.w-1)
            
            #=@spawn=# apply_tiny_Q_with_gap_left!(qr′, A′, τs′, merged_block_size)
            
            τ_offset += p.w
        end

        merged_block_size *= 2
    end
end

function apply_tsqr_right!(qr, A, τs, p::Panel)
    m, n = size(qr)

    @assert n == p.w

    num_panels_vertically = ÷(m, p.h, RoundUp)

    # Keep the τ offset conveniently here, cause
    # we need that when walking up the tree flater.
    τ_offset = 1

    # First standard QR on every panel.
    for b = OneTo(num_panels_vertically)
        b_start = (b - 1) * p.h + 1
        b_end = min(b_start + p.h - 1, m)
        τs′ = view(τs, τ_offset:τ_offset+p.w-1)

        qr′ = view(qr, b_start:b_end, :)
        A′ = view(A, :, b_start:b_end)

        #=@spawn=# apply_tiny_Q_right!(qr′, A′, τs′)

        τ_offset += p.w
    end

    # Then we walk up the tree and do QR on upper triangular blocks.
    tree_height = exponent_of_2_greater_than_or_equal_to(num_panels_vertically)

    # we're combining two panels of height `merged_block_size`
    merged_block_size = p.h

    # we've already done the leaf nodes, so - 1.
    for level = OneTo(tree_height)
        num_panels_vertically = ÷(m, merged_block_size, RoundUp)

        # combine all pairs of panels.
        for b = OneTo(num_panels_vertically ÷ 2)
            b_start = (b - 1) * 2merged_block_size + 1
            b_end = min(b_start + 2merged_block_size - 1, m)

            qr′ = view(qr, b_start:b_end, :)
            A′ = view(A, :, b_start:b_end)
            τs′ = view(τs, τ_offset:τ_offset+p.w-1)
            
            #=@spawn=# apply_tiny_Q_with_gap_right!(qr′, A′, τs′, merged_block_size)
            
            τ_offset += p.w
        end

        merged_block_size *= 2
    end
end

### just a simple dense QR.

function block_qr!(A, τs)
    m, n = size(A)

    B = copy(A)

    steps = min(m - 1, n)

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(steps)
        x = view(A, k:m, k)
        τs[k] = reflector!(x)
        reflectorApplyLeft!(view(A, k:m, k+1:n), τs[k], x)
    end

    return A
end

### QR two stacked, upper triangular matrices. 

function merge_block_qr!(A, τs, v_block_size)
    m, n = size(A)

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(n)
        gap = v_block_size - k + 1
        to = min(m, k + v_block_size)
        x = view(A, k:to, k)
        τs[k] = reflector!(x, gap)
        reflectorApplyLeft!(view(A, k:to, k+1:n), τs[k], x, gap)
    end

    return A
end

### Applying Q from a tiny QR decomp.

function apply_tiny_Q_left!(qr, A, τs)
    m, n = size(qr)

    steps = min(m - 1, n)
    
    @inbounds for k = OneTo(steps)
        reflectorApplyLeft!(view(A, k:size(A, 1), :), τs[k], view(qr, k:m, k))
    end

    return A
end

function apply_tiny_Q_right!(qr, A, τs)
    m, n = size(qr)

    steps = min(m - 1, n)
    
    @inbounds for k = OneTo(steps)
        reflectorApplyRight!(view(A, :, k:size(A, 2)), τs[k], view(qr, k:m, k))
    end
    
    return A
end

function apply_tiny_Q_with_gap_left!(qr, A, τs, v_block_size)
    m, n = size(qr)

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(n)
        gap = v_block_size - k + 1
        to = min(m, k + v_block_size)
        x = view(qr, k:to, k)
        reflectorApplyLeft!(view(A, k:to, :), τs[k], x, gap)
    end

    return A
end

function apply_tiny_Q_with_gap_right!(qr, A, τs, v_block_size)
    m, n = size(qr)

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(n)
        gap = v_block_size - k + 1
        to = min(m, k + v_block_size)
        x = view(qr, k:to, k)
        reflectorApplyRight!(view(A, :, k:to), τs[k], x, gap)
    end

    return A
end

#### Computing and applying a single reflector

@inline function reflector!(x::AbstractVector, gap = 1)
    @inbounds begin
        n = length(x)
        n == 0 && return zero(eltype(x))
        ξ1 = x[1]
        normu = abs2(ξ1)
        for i = 1+gap:n
            normu += abs2(x[i])
        end
        iszero(normu) && return zero(eltype(x))
        normu = sqrt(normu)
        ν = copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        for i = 1+gap:n
            x[i] /= ξ1
        end
        ξ1/ν
    end
end

@inline function reflectorApplyLeft!(A::AbstractMatrix, τ::Number, x::AbstractVector, gap = 1)
    m, n = size(A)
    m == 0 && return A

    @inbounds for j = 1:n
        # dot
        vAj = A[1, j]
        for i = 1+gap:m
            vAj += x[i]'*A[i, j]
        end

        vAj = conj(τ)*vAj

        # ger
        A[1, j] -= vAj
        for i = 1+gap:m
            A[i, j] -= x[i]*vAj
        end
    end
    return A
end

@inline function reflectorApplyRight!(A::AbstractMatrix, τ::Number, x::AbstractVector, gap = 1)
    m, n = size(A)
    n == 0 && return A
    @inbounds for j = 1:m
        # dot
        vAj = A[j, 1]
        for i = 1+gap:n
            vAj += x[i]*A[j, i]
        end

        vAj = τ*vAj

        # ger
        A[j, 1] -= vAj
        for i = 1+gap:n
            A[j, i] -= x[i]'*vAj
        end
    end
    return A
end
