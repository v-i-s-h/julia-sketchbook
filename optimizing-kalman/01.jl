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
    λ = 1/100
    Q = 1e-20I
    R = 1e-2I

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

    # Simulation
    result = Vector{Float64}(undef, 60000)

    result[1] = tr(Pu)

    for k = 2:60000
        Pp = Fk_1 * Pu * Fk_1' + Q
        K = Pp * Hk' * pinv(R + Hk * Pp * Hk')
        Pu = (I - K * Hk) * Pp * (I - K * Hk)' + K * R * K'

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
