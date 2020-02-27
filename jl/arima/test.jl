using Turing, Distributions, StatsBase, DataFrames, CSV, HypothesisTests, LinearAlgebra
using MCMCChains, Plots, StatsPlots
using Random
using MLJ

Random.seed!(12);
# Turn off progress monitor.
Turing.turnprogress(false)
df = CSV.read("data_master_1.csv")
df[:sp_500]
s = df.sp_500
plot(s)


train, test = partition(1:nrow(df), 0.8)
plot(s[train])

ADFTest(s[train], Symbol("constant"), 5)
sdiff = diff(s[train])
plot(sdiff)
ADFTest(sdiff, Symbol("constant"), 5)

@model ARIMA110(x) = begin
    T = length(x)
    μ ~ Uniform(-10, 10)
    ϕ ~ Uniform(-1, 1)
    for t in 3:T
        val = μ +                      # Drift term.
              x[t-1] +                 # ARIMA(0,1,0) portion.
              ϕ * (x[t-1] - x[t-2]) # ARIMA(1,0,0) portion.
        x[t] ~ Normal(val, 1)
    end
end

@model ARIMA011(x) = begin
    T = length(x)
    # Set up error vector.
    ϵ = Vector(undef, T)
    x_hat = Vector(undef, T)
    θ ~ Uniform(-5, 5)
    # Treat the first x_hat as a parameter to estimate.
    x_hat[1] ~ Normal(0, 1)
    ϵ[1] = x[1] - x_hat[1]
    for t in 2:T
        # Predicted value for x.
        x_hat[t] = x[t-1] - θ * ϵ[t-1]
        # Calculate observed error.
        ϵ[t] = x[t] - x_hat[t]
        # Observe likelihood.
        x[t] ~ Normal(x_hat[t], 1)
    end
end

chain_ARIMA110 = sample(ARIMA110(s[train]), NUTS(10000, 200, 0.6) )
chain_ARIMA011 = sample(ARIMA011(s[train]), NUTS(5000, 200, 0.6) )
