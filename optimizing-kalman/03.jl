"""
Source: https://www.ronanarraes.com/2019/04/my-julia-workflow-readability-vs-performance/
"""

using LinearAlgebra

# Constants
const O3x3  = zeros(3, 3)
const O3x12 = zeros(3, 12)
const O12x18    = zeros(12, 18)

function kalman_filter()
    # Constants
    I6 = Matrix{Float64}(I, 6, 6)
    I18 = Matrix{Float64}(I, 18, 18)
    λ = 1/100
    Q = 1e-20I18
    R = 1e-2I6

    # Dynamic model
    wx = Float64[ 
        +0 -1 +0;
        +1 +0 +0;
        +0 +0 +0
    ]

    Ak_1 = [
        wx      -I      O3x12;
        O3x3    λ*I     O3x12;
            O12x18;
    ]

    Fk_1 = exp(Ak_1)

    # Measurement model
    sx = Float64[
        +0 +0 +0;
        +0 +0 -1;
        +0 +1 +0;
    ]

    Bx = Float64[
        +0 +0 -1;
        +0 +0 +0;
        +1 +0 +0;
    ]

    Hk = [
        sx      O3x3    sx      I       O3x3    O3x3;
        Bx      O3x3    O3x3    O3x3    Bx      I;
    ]

    # Kalman filter initialization
    Pu = Matrix{Float64}(I, 18, 18)
    Pp = similar(Pu)

    # Simulation
    result = Vector{Float64}(undef, 60000)

    result[1] = tr(Pu)

    # auxilary variables
    K = zeros(18, 6)
    aux1 = zeros(18, 18)
    aux2 = zeros(18, 18)
    aux3 = zeros(18, 6)
    aux4 = zeros(6, 6)
    aux5 = zeros(18, 18)
    aux6 = zeros(18, 18)


    @inbounds for k = 2:60000
        # Pp .= Fk_1 * Pu * Fk_1' .+ Q
        mul!(aux2, mul!(aux1, Fk_1, Pu), Fk_1')
        @. Pp = aux2 + Q

        # K .= Pp * Hk' * pinv(R .+ Hk * Pp * Hk')
        mul!(aux4, Hk, mul!(aux3, Pp, Hk'))
        mul!(K, aux3, pinv(R .+ aux4))

        # Pu = (I - K * Hk) * Pp * (I - K * Hk)' + K * R * K'
        mul!(aux1, K, Hk)
        @. aux2 = I18 - aux1
        mul!(aux6, mul!(aux5, aux2, Pp), aux2')
        mul!(aux5, mul!(aux3, K, R), K')
        @. Pu = aux5 + aux6

        result[k] = tr(Pu)
    end

    return result
end

#=
    Run as:
    using BenchmarkTools
    @btime kalman_filter();
=#

# r = kalman_filter()

using BenchmarkTools
display(@benchmark kalman_filter())
println()
