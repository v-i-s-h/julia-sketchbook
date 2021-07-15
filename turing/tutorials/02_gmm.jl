# Unsupervised Learning using Bayesian Mixture Models
# Source: https://turing.ml/dev/tutorials/01-gaussian-mixture-model/

using Distributions, StatsPlots, Random

##
# Set a random seed
Random.seed!(3)

# Construct 30 data points for each cluster
N = 30

# Parametes for each cluster, we assume that each cluster is Gaussian distributed
μs = [ -3.5, 0.0]

# Construct data points
x = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.0), N), hcat, 1:2)

# Visualization
scatter(x[1, :], x[2, :], legend=false, title="Synthetic dataset")

##
using Turing, MCMCChains

##
@model GaussianMixtureModel(x) = begin
    
    D, N = size(x)

    # Draw the parameters for cluster 1
    μ1 ~ Normal()

    # Draw the parameters for cluster 2
    μ2 ~ Normal()

    μ = [ μ1, μ2]

    w = [ 0.5, 0.5]

    # Draw assignments for each datum and generate it from a multivariate normal
    k = Vector{Int}(undef, N)
    for i ∈ 1:N
        k[i] ~ Categorical(w)
        x[:, i] ~ MvNormal([μ[k[i]], μ[k[i]]], 1.0)
    end

    return k
end

## 
gmm_model = GaussianMixtureModel(x)

##
gmm_sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ1, :μ2))
tchain = mapreduce(c -> sample(gmm_model, gmm_sampler, 100), chainscat, 1:3)

##
ids = findall(map(name -> occursin("μ", string(name)), names(tchain)))
p = plot(tchain[:, ids, :], legend=true, labels=["μ1" "μ2"], colordim=:parameter)

##
tchain = tchain[:, :, 1]

##
# Helper function used for visualizing the density region
function predict(x, y, w, μ)
    # Use log-sum-exp trick for numeric stability
    return Turing.logaddexp(
        log(w[1]) + logpdf(MvNormal([μ[1], μ[1]], 1.0), [x, y]),
        log(w[2]) + logpdf(MvNormal([μ[2], μ[2]], 1.0), [x, y])
    )
end

##
contour( -10:0.10:+10.0, -10:0.10:+10.0,
    (x, y) -> predict(x, y, [0.5, 0.5], [mean(tchain[:μ1]), mean(tchain[:μ2])])
)
scatter!(x[1, :], x[2, :], legend=false, title="Synthetic Dataset")

##
assignments = mean(MCMCChains.group(tchain, :k)).nt.mean
scatter(x[1, :], x[2, :],
    legend=false,
    title="Assignments on Synthetic Dataset",
    zcolor=assignments)
