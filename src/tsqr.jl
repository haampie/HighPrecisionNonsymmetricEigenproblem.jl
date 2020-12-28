using Base: OneTo

using Base.Threads: @spawn
using Base: @sync

function to_band!(A::AbstractMatrix{T}, block_size = 32) where {T}
    # iterate in blocks.

    m = size(A, 1)
    blocks = ÷(m, block_size, RoundUp)
  
    offset_col = 1
    offset_row = 1 + block_size

    τs = Vector{T}(undef, m)

    for block = OneTo(blocks)
        cols_to_band!(A, offset_row, offset_col, block_size, τs)
        offset_row += block_size
        offset_col += block_size
    end

    return A
end

function cols_to_band!(A, offset_row, offset_col, block_size, τs)
    # The part in which we do a tall and skinny QR
    A_qr = view(A, offset_row:size(A, 1), offset_col:min(size(A, 2), offset_col+block_size-1))

    # The part to which we apply the tall and skinny QR from the left
    A_left = view(A, offset_row:size(A, 1), offset_col+block_size:size(A, 2))

    # The part to which we apply the tall and skinny QR from the right.
    A_right = view(A, :, offset_row:size(A, 1))

    #=@sync=# for block_start = 1:block_size:size(A_qr, 1)
        let
            block_end = min(block_start + block_size - 1, size(A_qr, 1))
            range = block_start:block_end
            A_qr′ = view(A_qr, range, :)
            τs′ = view(τs, range)
            #=@spawn=# block_qr!(A_qr′, τs′)
        end
    end

    #=@sync=# for block_start = 1:block_size:size(A_qr, 1)
        let
            block_end = min(block_start + block_size - 1, size(A_qr, 1))
            range = block_start:block_end
            A_qr′ = view(A_qr, range, :)
            A_left′ = view(A_left, range, :)
            τs′ = view(τs, range)
            #=@spawn=# apply_left!(A_qr′, A_left′, τs′)
        end
    end

    #=@sync=# for block_start = 1:block_size:size(A_qr, 1)
        let
            block_end = min(block_start + block_size - 1, size(A_qr, 1))
            range = block_start:block_end
            A_qr′ = view(A_qr, range, :)
            A_right′ = view(A_right, :, range)
            τs′ = view(τs, range)
            #=@spawn=# apply_right!(A_qr′, A_right′, τs′)
        end
    end

    # Pair-wise reduce to single QR.
    merged_block_size = block_size
    while true
        # If there's only one block left, stop.
        nblocks = ÷(size(A_qr, 1), merged_block_size, RoundUp)
        nblocks ≤ 1 && break

        #=@sync=# for block_start = 1:2merged_block_size:size(A_qr, 1)
            let
                # If only a single block or less fits, stop.
                block_start + merged_block_size - 1 ≥ size(A_qr, 1) && break
    
                block_end = min(block_start + 2merged_block_size - 1, size(A_qr, 1))
                range = block_start:block_end
                A_qr′ = view(A_qr, range, :)
                τs′ = view(τs, range)
                #=@spawn=# merge_block_qr!(A_qr′, τs′, merged_block_size)
            end
        end

        #=@sync=# for block_start = 1:2merged_block_size:size(A_qr, 1)
            let
                # If only a single block or less fits, stop.
                block_start + merged_block_size - 1 ≥ size(A_qr, 1) && break

                block_end = min(block_start + 2merged_block_size - 1, size(A_qr, 1))
                range = block_start:block_end
                A_qr′ = view(A_qr, range, :)
                A_left′ = view(A_left, range, :)
                τs′ = view(τs, range)
                #=@spawn=# apply_left_with_gap!(A_qr′, A_left′, τs′, merged_block_size)
            end
        end

        #=@sync=# for block_start = 1:2merged_block_size:size(A_qr, 1)
            let
                # If only a single block or less fits, stop.
                block_start + merged_block_size - 1 ≥ size(A_qr, 1) && break

                block_end = min(block_start + 2merged_block_size - 1, size(A_qr, 1))
                range = block_start:block_end
                A_qr′ = view(A_qr, range, :)
                A_right′ = view(A_right, :, range)
                τs′ = view(τs, range)
                #=@spawn=# apply_right_with_gap!(A_qr′, A_right′, τs′, merged_block_size)
            end
        end

        merged_block_size *= 2
    end

    A
end

function block_qr!(A, τs)
    m, n = size(A)

    steps = min(m, n) - 1

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(steps)
        x = view(A, k:m, k)
        τs[k] = reflector!(x)
        reflectorApplyLeft!(view(A, k:m, k+1:n), τs[k], x)
    end

    return A
end

function apply_left!(qr, A, τs)
    m, n = size(qr)

    steps = min(m, n) - 1
    
    @inbounds for k = OneTo(steps)
        reflectorApplyLeft!(view(A, k:size(A, 1), :), τs[k], view(qr, k:m, k))
    end

    return A
end

function apply_right!(qr, A, τs)
    m, n = size(qr)

    steps = min(m, n) - 1
    
    @inbounds for k = OneTo(steps)
        reflectorApplyRight!(view(A, :, k:size(A, 2)), τs[k], view(qr, k:m, k))
    end
    
    return A
end

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

function apply_left_with_gap!(qr, A, τs, v_block_size)
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

function apply_right_with_gap!(qr, A, τs, v_block_size)
    m, n = size(qr)

    # Create a bunch of reflectors and save the norm value (τ)
    @inbounds for k = OneTo(n)
        gap = v_block_size - k + 1
        to = min(m, k + v_block_size)
        x = view(qr, k:to, k)
        reflectorApplyRight!(view(A, :, k:to), τs[k], x, gap)
        qr[k+1:to, k] .= 0
    end

    return A
end

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
