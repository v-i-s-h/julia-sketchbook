# Source:

using Lux, Random

##
rng = Random.default_rng()
Random.seed!(rng, 0)
Random.TaskLocalRNG()

##
x = [1, 2, 3]

x = [
    1 2;
    3 4;
    5 6
]

x = rand(rng, 5, 3)

x = rand(rng, Float32, 5, 3)

length(x)
size(x)
x[2, 3]

##

using ForwardDiff, Zygote, AbstractDifferentiation

##
f(x) = x' * x / 2
∇f(x) = x
v = randn(rng, Float64, 4)

##
println("Actual gradient = ", ∇f(v))
println("Compute (Zygote, Rev AD) = ", AD.gradient(AD.ZygoteBackend(), f, v)[1])
println("Computed (Fwd AD) = ", AD.gradient(AD.ForwardDiffBackend(), f, v)[1])

# JVP
f(x) = x .* x ./ 2
x = randn(Float32, 5)
v = ones(Float32, 5)

pf_f = AD.value_and_pushforward_function(AD.ForwardDiffBackend(), f, x)

val, jvp = pf_f(v)
println("Computed = ", val)
println("JVP = ", jvp[1])

# VJP
pb_f = AD.value_and_pullback_function(AD.ZygoteBackend(), f, x)

val, vjp = pb_f(v)
println("Computed value : ", val)
println("VJP: ", vjp[1])

## Linear regression problem
model = Dense(10 => 5)
rng = Random.default_rng()
Random.seed!(rng, 0)
Random.TaskLocalRNG()

##
ps, st = Lux.setup(rng, model)
ps = ps |> Lux.ComponentArray

##
n_samples = 20
x_dim = 10
y_dim = 5

##
W = randn(rng, Float32, y_dim, x_dim)
b = randn(rng, Float32, y_dim)

##
x_samples = randn(rng, Float32, x_dim, n_samples)
y_samples = W * x_samples .+ b .+ 0.01f0 .* randn(rng, Float32, y_dim, n_samples)

##
using Optimisers

opt = Optimisers.Descent(0.010f0)

opt_state = Optimisers.setup(opt, ps)

##
mse(model, ps, st, X, y) = sum(abs2, model(X, ps, st)[1] .- y)
mse(weights, bias, X, y) = sum(abs2, weights * X .+ bias .- y)
loss_function(ps, X, y) = mse(model, ps, st, X, y)

println("Loss with W & b = ", mse(W, b, x_samples, y_samples))

##
for i ∈ 1:100
    global ps, st, opt_state
    # gs = gradient(loss_function, ps, x_samples, y_samples)[1]
    gs = zeros(size(ps)...)
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    if i % 10 == 1 || i == 100
        println("Loss value @ $i = ", mse(model, ps, st, x_samples, y_samples))
    end
end

##




