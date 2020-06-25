#=
    To explore optimized implementations for data generation code used in 01.jl
=#

import LinearAlgebra: mul!, dot
using TimerOutputs

## Version 1 - Direct Multiplication
# pre-allocated simple version
function func1!(y, A, x)
    y .= A * x
end

# Simple version for humans
function func1(A, x)
    y = Vector(undef, size(A, 1))
    func1!(y, A, x)
end

## Version 2 - With eachindex
# Pre-allocated
function func2!(y, A, x)
    for i in eachindex(y)
        y[i] = dot(A[i, :], x)
    end
end

# for humans
function func2(A, x)
    y = Vector(undef, size(A, 1))
    func2!(y, A, x)
end

## Version 3 - for with size()
# pre-allocated
function func3!(y, A, x)
    n = size(y, 1)
    for i = 1:n
        y[i] = dot(A[i, :], x)
    end
end

# for humans
function func3(A, x)
    y = Vector(undef, size(A, 1))
    func3!(y, A, x)
end

## Version 4 - with inbounds
function func4!(y, A, x)
    n = length(y)
    @inbounds for i = 1:n
        @timeit "dot + slice" begin
            @timeit "slice" t = A[i, :]
            @timeit "dot" y[i] = dot(t, x)
        end
    end
end

# for humans
function func4(A, x)
    y = Vector(undef, size(A, 1))
    func4!(y, A, x)
end

## Version 5 - with inbounds and views
function func5!(y, A, x)
    n = length(y)
    @inbounds for i = 1:n
        @timeit "dot + @view" begin
            @timeit "@view" t = @view A[i, :]
            @timeit "dot()" y[i] = dot(t, x)
        end
    end
end

# for humans
function func5(A, x)
    y = Vector(undef, size(A, 1))
    func5!(y, A, x)
end

# # Benchmarking
function benchmark()
    
    ndims = 100
    nsamples = 50000
    A = rand(nsamples, ndims)
    x = rand(ndims)

    y = Vector(undef, size(A, 1))

    reset_timer!()

    # run all tests 100 times
    for i = 1:100
        ## 1
        @timeit "Direct Multiplication" begin
            @timeit "func1" begin
                y = func1(A, x)
            end
            
            @timeit "func1!" begin
                @timeit "allocate" y = Vector(undef, size(A, 1))
                @timeit "func1!" func1!(y, A, x)
            end
            
        end

        ## 2
        @timeit "eachindex()" begin
            @timeit "func2" begin
                y = func2(A, x)
            end

            @timeit "func2!" begin
                @timeit "allocate" y = Vector(undef, size(A, 1))
                @timeit "func2!" func2!(y, A, x)
            end
        end

        ## 3
        @timeit "length()" begin
            @timeit "func3" begin
                y = func3(A, x)
            end

            @timeit "func3!" begin
                @timeit "allocate" y = Vector(undef, size(A, 1))
                @timeit "func3!" func3!(y, A, x)
            end
        end

        ## 4
        @timeit "length()+@inbounds" begin
            @timeit "func4" begin
                y = func4(A, x)
            end

            @timeit "func4!" begin
                @timeit "allocate" y = Vector(undef, size(A, 1))
                @timeit "func4!" func4!(y, A, x)
            end
        end

        ## 5
        @timeit "length+@inbounds+@view" begin
            @timeit "func5" begin
                y = func5(A, x)
            end

            @timeit "func5!" begin
                @timeit "allocate" y = Vector(undef, size(A, 1))
                @timeit "func5!" func5!(y, A, x)
            end
        end
    end
    print_timer()
end

# benchmark()
# benchmark()

nothing