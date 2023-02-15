# Test for struct functions

##
struct RandomModel{T}
    a::T
    b::T
end

##
function RandomModel(a::T) where T
    RandomModel(a, 2a)
end

##
function (m::RandomModel{T})() where {T<:AbstractFloat}
    println("FP:: a = $(m.a), b = $(m.b)")
end

##

a = RandomModel(4, 6)
b = RandomModel(4.0, 6.0)

a()

b()

##