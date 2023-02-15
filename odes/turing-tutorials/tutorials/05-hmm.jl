# Bayesian Hidden Markov models

using Turing, StatsPlots, Random

Turing.setprogress!(false)
Turing.setadbackend(:forwarddiff)
Random.seed!(123);

##
y = [ 1.0 * ones(6);
      2.0 * ones(6);
      3.0 * ones(6);
      2.0 * ones(6);
      1.0 * ones(6); ]
N = length(y)
K = 3

plot(y; xlim=(0, N), ylim=(-1, 5), size=(500, 250))

##
@model function BayesHmm(y, K)
    N = length(y) # Observation length
    s = tzeros(Int, N) # State sequence
    m = Vector(undef, K) # emission matrix
    T = Vector{Vector}(undef, K) # transition matrix

    for i ∈ 1:K
        T[i] ~ Dirichlet(ones(K) / K)
        m[i] ~ Normal(i, 0.5)
    end

    # Observe each point of the input
    s[1] ~ Categorical(K)
    y[1] ~ Normal(m[s[1]], 0.1)

    for i ∈ 2:N
        s[i] ~ Categorical(vec(T[s[i - 1]]))
        y[i] ~ Normal(m[s[i]], 0.1)
    end

    return y
end

##
g = Gibbs(HMC(0.01, 50, :m, :T), PG(120, :s))
chn = sample(BayesHmm(y, 3), g, 1000)

##
m_set = MCMCChains.group(chn, :m).value
s_set = MCMCChains.group(chn, :s).value

Ns = 1:length(chn)

animation = @gif for i ∈ Ns
    m = m_set[i, :]
    s = Int.(s_set[i, :])
    emissions = m[s]

    p = plot(
        y;
        chn=:red,
        size=(500, 250),
        xlabel="Time",
        ylabel="State",
        legend=:topright,
        label="True data",
        xlim=(0, 30),
        ylim=(-1, 5),
    )
    plot!(emissions; color=:blue, label="Sample $i")
end every 3

##


