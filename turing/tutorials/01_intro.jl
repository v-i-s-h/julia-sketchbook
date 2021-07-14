# Introduction to Turing
# Source: https://turing.ml/dev/tutorials/00-introduction/

# Use Base modules
using Random

# Load a plotting library
using Plots

# Load the distributions library
using Distributions

# Set the true probability of heads in a coin
p_true = 0.5

# Iterate from having seen 0 observations to 100 observations
Ns = 0:100

##
# Draw data from a Bernoulli distribution, i.e. draw heads or tails
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# Here's what the first five coin flips look like:
data[1:5]

##
# Our prior belief about the probability of heads in a coin toss
prior_belief = Beta(1, 1)

##
# Import StatsPlots for animating purposes
using StatsPlots

# Make animation
animation = @gif for (i, N) in enumerate(Ns)

    # Count the number of heads and tails
    heads = sum(data[1:i-1])
    tails = N - heads

    # Update our prior belief in closed form
    updated_belief = Beta(prior_belief.α + heads, prior_belief.β + tails)

    # Plotting
    plot(updated_belief,
        size(500, 250),
        title="Updated belief after $N observations",
        xlabel="Probability of heads",
        ylabel="",
        legend=nothing,
        xlim=(0, 1),
        fill=0, α=0.3, w=3)
    vline!([p_true])
end

##
using Turing, MCMCChains

##
@model coinflip(y) = begin
    
    # our prior belief about the probability of heads in a coin
    p ~ Beta(1, 1)
    
    # the number of observations
    N = length(y)
    for n in 1:N
        y[n] ~ Bernoulli(p)
    end
end

##
# Settings of the Hamiltonian Monte Carlo (HMC) sampler
iterations = 1000
ϵ = 0.05
τ = 10

# Start sampling
chain = sample(coinflip(data), HMC(ϵ, τ), iterations, progess=false)

##
# Construct summary of the sampling process for the parameter p
p_summary = chain[:p]
plot(p_summary, seriestype=:histogram)

##
# Compute the posterior distribution in closed form
N = length(data)
heads = sum(data)
updated_belief = Beta(prior_belief.α + heads, prior_belief.β + N - heads)

# Visualize a blue dennsity plot of the approximate posterior distribution
p = plot(p_summary, seriestype=:density, xlim=(0, 1), legend=:best, w=2, c=:blue)

# Visualize a green density plot of posterior distribution in closed form
plot!(p,
      range(0, stop=1, length=100),
      pdf.(Ref(updated_belief), range(0, stop=1, length=100)),
      xlabel="probability of heads", ylabel="", title="",
      xlim=(0, 1), label="Closed form",
      fill=0, α=0.3, w=3, c=:lightgreen)

# Visualize the true probabilty of heads in red
vline!(p, [p_true], label="True probability", c=:red)
