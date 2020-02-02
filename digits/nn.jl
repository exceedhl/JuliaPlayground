using Distributions
using LineSearches, Optim

include("util.jl")

mutable struct BasicNN
    Ln::Array{Unsigned, 1}
    L
    𝚯
    λ::Float64

    function BasicNN(Ln, λ=1)
        L = []
        for i in 1:(length(Ln) - 1)
            push!(L, [Ln[i+1], Ln[i] + 1])
        end
        𝚯 = init_θ(L)
        new(Ln, L, 𝚯, λ)
    end
end

function init_θ(L)
    ϵ = 0.12
    init_θ = rand(Distributions.Uniform(-ϵ, ϵ), mapreduce(prod, +, L))
    vector_to_array(init_θ, L)
end

sigmoid(z) = 1.0 ./ (1.0 + exp(-z))

function H(x, 𝚯) # refactor to support more layers
    θ1, θ2 = 𝚯
    a1 = x
    a2 = sigmoid.(θ1 * [1; x])
    a3 = sigmoid.(θ2 * [1; a2])
    [a1, a2, a3]
end

function cost(a, y)
    -log.(a[end])' * y - log.(1 .- a[end])' * (1 .- y)
end

function J(nn::BasicNN, X_train, Y_train)
    θ1, θ2 = nn.𝚯
    C = 0
    for i = 1:m
        x = X_train[i, :]
        y = Y_train[i]
        a = H(x, nn.𝚯)
        C += cost(a, y)
    end
    C = C / m + (sum(θ1[:, 2:end] .^ 2) + sum(θ2[:, 2:end] .^ 2)) * λ / (2 * m)
end

function Δ𝚯(a, y, 𝚯) # refactor to support more layers
    a1, a2, a3 = a
    θ1, θ2 = 𝚯
    δ3 = (a3 - y)
    δ2 = (θ2'*δ3)[2:end] .* (a2 .* (1 .- a2))
    [δ2 * [1; a1]', δ3 * [1; a2]']
end

function ΔJ(nn::BasicNN, X_train, Y_train)
    Δ𝚯s = map(l->zeros(l...), nn.L)
    for i = 1:m
        x = X_train[i, :]
        y = Y_train[i]
        a = H(x, nn.𝚯)
        Δ𝚯s += Δ𝚯(a, y, nn.𝚯)
    end
    Δ𝚯s /= m
    for i = 1:length(Δ𝚯s)
        Δ𝚯s[i][:, 2:end] += λ * nn.𝚯[i][:, 2:end] / m
    end
    Δ𝚯s
end

function train!(nn::BasicNN, X_train, Y_train; λ = 1, iterations = 50)
    # update nn's λ if provided
    nn.λ = λ

    # use Optim to find minimum 𝚯
    function f(θ_vec)
        nn.𝚯 = vector_to_array(θ_vec, nn.L)
        J(nn, X_train, Y_train)
    end

    function g!(storage, θ_vec)
        nn.𝚯 = vector_to_array(θ_vec, nn.L)
        g = array_to_vector(ΔJ(nn, X_train, Y_train))
        for i = 1:length(storage)
            storage[i] = g[i]
        end
    end

    @time @show result = Optim.optimize(
        f,
        g!,
        array_to_vector(nn.𝚯),
        LBFGS(m=5, alphaguess=LineSearches.InitialHagerZhang(), linesearch=LineSearches.MoreThuente()),
        # LBFGS(m=20, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe()),
        Optim.Options(iterations = iterations, show_trace=true, show_every=1)
        # ConjugateGradient()
        # ConjugateGradient(eta=0.1, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe())
    )
    𝚯_min = vector_to_array(result.minimizer, nn.L)
    nn.𝚯 = 𝚯_min
end

function predict(nn::BasicNN, x)
    H(x, nn.𝚯)[end]
end
