# Unspervised learning using Bayesian Mixture Models

using Distributions
using FillArrays
using StatsPlots;

using LinearAlgebra
using Random

using Turing

Random.seed!(3); # For reproducibility

## Generate data
w = [0.5, 0.5]
μ = [-3.5, 0.5]
mixturemodel = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ ∈ μ], w)

N = 60
x = rand(mixturemodel, N);

scatter(x[1, :], x[2, :]; legend=false, title="Synthetic dataset")

##
@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K = 2 clusters from a N(0, 1)
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution
    # with parameter αₖ = 1
    w ~ Dirichlet(K, 1.0)

    # Construct categorical distribution assignments
    distribution_assignments = Categorical(w)

    # construct categorical normal dsitributions of each cluster
    D, N = size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ ∈ μ]

    # Draw assignments for each datum and generate it from the multivariate
    # normal distribution
    k = Vector{Int}(undef, N)
    for i ∈ 1:N
        k[i] ~ distribution_assignments
        x[:, i] ~ distribution_clusters[k[i]]
    end

    return k
end

model = gaussian_mixture_model(x);

##
sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ, :w))
nsamples = 100
nchains = 3
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains)

##
plot(chains[["μ[1]", "μ[2]"]]; colordim=:parameter, legend=true)

##
plot(chains[["w[1]", "w[2]"]]; colordim=:parameter, legend=true)

##
chain = chains[:, :, 1]

## Visualize 
μ_mean = [mean(chain, "μ[$i]") for i in 1:2]
w_mean = [mean(chain, "w[$i]") for i in 1:2]
mixturemodel_mean = MixtureModel([MvNormal(Fill(μₖ, 2), I) for μₖ ∈ μ_mean], w_mean)

contour(
    range(-7.5, 3; length=1000),
    range(-6.5, 3; length=1000),
    (x, y) -> logpdf(mixturemodel_mean, [x, y]);
    widen=false
)
scatter!(x[1, :], x[2, :]; legend=false, title="Synthetic Dataset")

##
