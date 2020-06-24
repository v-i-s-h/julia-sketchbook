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

    @inbounds for k = 2:60000
        Pp .= Fk_1 * Pu * Fk_1' .+ Q
        K .= Pp * Hk' * pinv(R .+ Hk * Pp * Hk')
        aux1 = I18 .- K * Hk
        Pu = aux1 * Pp * aux1' .+ K * R * K'

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
