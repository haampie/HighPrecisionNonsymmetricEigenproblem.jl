using HighPrecisionNonsymmetricEigenproblem
using Test
using MultiFloats
using LinearAlgebra

using HighPrecisionNonsymmetricEigenproblem: pack!, do_qr!, unpack!, apply_tiny_Q_left!

@testset "HighPrecisionNonsymmetricEigenproblem.jl" begin
    T = Float64x2
    Q = rand(T, 32, 4)
    t = rand(T, 4)
    A = rand(T, 32, 9968)

    Q_packed, t_packed, A_packed = pack!(Q, t, A)
    do_qr!(Q_packed, t_packed, A_packed, T)
    A_fast_kernel = unpack!(similar(A), A_packed)
    A_slow_kernel = apply_tiny_Q_left!(Q, copy(A), t)

    @test norm(A_fast_kernel - A_slow_kernel) < 1e-20
end
