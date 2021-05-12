# Julia script for investigating the effect of temparature on softmax predictions

using Statistics
using Plots

# Define softmax function
softmax(logit::Vector{Float64}, τ::Float64=1.0) = exp.(logit ./ τ) ./ sum(exp.(logit./τ))
# Define entropy function
entropy(scores) = sum(-scores .* log2.(scores))

# Experiment configuration
K = [2, 5, 10, 20, 50, 100] # Number of classes
N = 1000 # Number of samples
T = -2.0:0.25:2.0

# Create a figure
fig = plot(title="Effect of τ")

# For each classes
for k ∈ K
    logits =[randn(k) for i ∈ 1:N]

    results = Dict{Float64, Vector{Float64}}()

    for τ ∈ T
        scores = map(l -> softmax(l, exp10(τ)), logits)
        results[τ] = map(entropy, scores)
    end

    μ = mean.([results[τ] for τ ∈ T])
    σ = std.([results[τ] for τ ∈ T])

    # Plot results
    plot!(fig, exp10.(T), μ, gird=false, ribbon=σ, fillalpha=0.5, label="K = $k")
end

plot!(fig, xaxis=("τ", exp10.(T), :log10), yaxis=("Entropy (bits)"), legend=:topleft)

