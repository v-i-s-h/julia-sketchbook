# Convert a 3D array to 2D array with elements as vector
# Source: https://discourse.julialang.org/t/convert-a-3d-array-to-2d-array-of-vectors/65755/3

using BenchmarkTools


# Using StaticArray, by @sdanisch
using StaticArrays

function method1(X)
    ax = axes(X)

    map(CartesianIndices((ax[1], ax[2]))) do i
        x, y = Tuple(i)
        SVector(ntuple(z -> X[x, y, z], 2))
    end
end


# Using TensorCast, by @rafael.guerra
using TensorCast

function method2(X)
    @cast out[i, j]{k} := X[i, j, k]
end


# Using StaticArrays and StructArrays, by @piever
using StaticArrays, StructArrays

function method3(X::AbstractArray{T, N}) where {T, N}
    return StructArray{SVector{2, T}}(X, dims=N)
end

### Benchmarking
X = rand(Int8, 50, 50, 2);

b1 = @benchmark method1($X)

b2 = @benchmark method2($X)

b3 = @benchmark method3($X)

