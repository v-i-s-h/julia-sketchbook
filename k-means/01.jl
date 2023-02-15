#=
Analyze k-Mean clustering algorithm and see how the sum-of-squared-distance loss
change with number of clusters
=#

##
using Plots
using Clustering
using Random
using Statistics
using LinearAlgebra
# theme(:dark)

## Generate data
function generate_data(;k=4, nk=25)
    l = 5.0 # Limits
    d = 2 # dimension

    # Cluster centers
    μ = l .* randn(d, k)
    # μ = l .* reduce(
    #     hcat,
    #     LinRange(0, 2π, k+1) .|> x -> [sin(x), cos(x)]
    # )[:, 1:end-1]


    # For each cluster center, generate data points
    x = hcat(
        map(_μ -> randn(d, nk) .+ _μ, eachcol(μ))...
    )

    # Cluster membership
    η = repeat(collect(1:k), inner=nk)

    # Shuffle data points
    shuffle_index = Vector(1:k*nk) |> shuffle
    x = x[:, shuffle_index]
    permute!(η, shuffle_index)

    return x, μ, η
end

##
k = 5
x, μ, η = generate_data(k=k, nk=50)

fig = plot(size=(640, 640))
scatter!(x[1, :], x[2, :], label=nothing)
scatter!(μ[1, :], μ[2, :], markershape=:circle, markersize=10, label=nothing)

## Perform clustering using actual number of cluster centers
R = kmeans(x, k)

## Compute the ss cost
ss = sum(
    (x .- R.centers[:, assignments(R)]) .^ 2,
    dims=1
)

## Perform clustering with different number of centers and plot cost
n_trials = 10 # Number of trials
n_centers = Vector(1:10k)

ss_costs = zeros(length(n_centers), n_trials) # matrix to store all results
for (_i, _k) ∈ enumerate(n_centers)
    for _j ∈ range(1, n_trials)
        # Perform clustering
        _R = kmeans(x, _k)
        # Compute SS cost
        _ss = sum(_R.costs)
        ss_costs[_i, _j] = _ss
    end
end

## Plot results
ss_mean = mean(ss_costs, dims=2)
ss_err = std(ss_costs, dims=2) / √n_trials

## Compute lower bound from zha2001spectral
H = x' * x # Gram matrix for data
tr_H = sum(abs2, x) # Trace of H is equal to the squared-sum of data points
λₕ = eigen(H).values |> reverse # Eigen values, in decending order

# (4)
ss_lower_bound = tr_H .- cumsum(λₕ[1:n_centers[end]]) # Compute for given

##
fig = plot(size=(720, 360))
plot!(n_centers, ss_mean, label=nothing)
plot!(n_centers, ss_lower_bound, label="zha2001spectral")
yaxis!(:log)
xaxis!("k")
