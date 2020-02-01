using Plots, Images, CSV, DataFrames
using Distributions
using Optim
using Test

d = CSV.read("./digits/train.csv")
X = convert(Array, d[!, 2:end] ./ 255)

function transform_digit_to_bitvec(digit, levels)
    vec = zeros(levels)
    setindex!(vec, 1, digit + 1)
    return vec
end

function transform_bitvec_to_digit(vec)
    findmax(vec)[2] - 1
end

Y = map(x->transform_digit_to_bitvec(x, 10), d[!, 1])

m = 40000
X_train = X[1:m, :]
Y_train = Y[1:m]

sigmoid(z) = 1.0 ./ (1.0 + exp(-z))

function H(x, 𝚯)
    θ1, θ2 = 𝚯
    a1 = x
    a2 = sigmoid.(θ1 * [1; x])
    a3 = sigmoid.(θ2 * [1; a2])
    [a1, a2, a3]
end

function cost(a, y)
    a1, a2, a3 = a
    -log.(a3)' * y - log.(1 .- a3)' * (1 .- y)
end

function J(𝚯)
    θ1, θ2 = 𝚯
    C = 0
    for i = 1:m
        x = X_train[i, :]
        y = Y_train[i]
        a = H(x, 𝚯)
        C += cost(a, y)
    end
    C = C / m + (sum(θ1[:, 2:end] .^ 2) + sum(θ2[:, 2:end] .^ 2)) * λ / (2 * m)
end

function Δ𝚯(a, y, 𝚯)
    a1, a2, a3 = a
    θ1, θ2 = 𝚯
    δ3 = (a3 - y)
    δ2 = (θ2'*δ3)[2:end] .* (a2 .* (1 .- a2))
    [δ2 * [1; a1]', δ3 * [1; a2]']
end

function ΔJ(𝚯)
    Δ𝚯s = map(θ->zeros(size(θ)), 𝚯)
    for i = 1:m
        x = X_train[i, :]
        y = Y_train[i]
        a = H(x, 𝚯)
        Δ𝚯s += Δ𝚯(a, y, 𝚯)
    end
    Δ𝚯s /= m
    for i = 1:length(Δ𝚯s)
        Δ𝚯s[i][:, 2:end] += λ * 𝚯[i][:, 2:end] / m
    end
    Δ𝚯s
end

function predict(x, 𝚯)
    a1, a2, a3 = H(x,𝚯)
    transform_bitvec_to_digit(a3)
end

L1 = 784
L2 = 50
L3 = 10
λ = 1

ϵ = 0.12
init_θ1 = reshape(
    rand(Distributions.Uniform(-ϵ, ϵ), (L1 + 1) * L2),
    L2,
    (L1 + 1),
) # L2 x L1
init_θ2 = reshape(
    rand(Distributions.Uniform(-ϵ, ϵ), (L2 + 1) * L3),
    L3,
    (L2 + 1),
) # L3 x L2


# use Optim to find minimum 𝚯
function array_to_vector(arr)
    mapreduce(x -> reshape(x, length(x)), vcat, arr, init = Float64[])
end

function vector_to_array(v, dims = [(L2, (L1 + 1)), (L3, (L2 + 1))])
    result = []
    i = 1
    for dim in dims
        push!(result, reshape(v[i:i+dim[1]*dim[2]-1], dim...))
        i += prod(dim)
    end
    result
end

@test array_to_vector([init_θ1, init_θ2]) ==
    array_to_vector(vector_to_array(array_to_vector([init_θ1, init_θ2])))

function f(θ_vec)
    J(vector_to_array(θ_vec))
end

function g!(storage, θ_vec)
    g = array_to_vector(ΔJ(vector_to_array(θ_vec)))
    for i = 1:length(storage)
        storage[i] = g[i]
    end
end

#=
using LineSearches
@time @show result = Optim.optimize(
    f,
    g!,
    array_to_vector([init_θ1, init_θ2]),
    LBFGS(m=5, alphaguess=LineSearches.InitialHagerZhang(), linesearch=LineSearches.MoreThuente()),
    # LBFGS(m=20, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe()),
    Optim.Options(iterations = 10, show_trace=true, show_every=1)
    # ConjugateGradient()
    # ConjugateGradient(eta=0.1, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe())
)
𝚯_min = vector_to_array(result.minimizer)

X_test = X[40001:42000, :]
Y_test = Y[40001:42000]
y_pred = [predict(X_test[i, :], 𝚯_min) for i in 1:length(Y_test)]
@show mean(y_pred .!= d[40001:42000, 1])
=#

# NLopt
# function f(x, grad)
#     grad = ΔJ(x)
#     return J(x)
# end
#
# opt = Opt(:LD_MMA, (length(init_θ1) + length(init_θ2)))
# opt.min_objective = f
#
# (minf,minx,ret) = optimize(opt, array_to_vector([init_θ1, init_θ2]))
#

# check gradient
L1 = 3
L2 = 3
L3 = 3
m = 5
X_train = reshape(rand(m*L1), m, L1)
Y_train = map(x->transform_digit_to_bitvec(x, 3), rand(0:2, m))
init_θ1 = reshape(
    rand(Distributions.Uniform(-ϵ, ϵ), (L1 + 1) * L2),
    L2,
    (L1 + 1),
) # L2 x L1
init_θ2 = reshape(
    rand(Distributions.Uniform(-ϵ, ϵ), (L2 + 1) * L3),
    L3,
    (L2 + 1),
) # L3 x L2

function checkGradient(𝚯)
    ϵ = 1e-4
    numgrad = zeros(size(𝚯))
    perturb = zeros(size(𝚯))
    for i in 1:length(𝚯)
        perturb[i] = ϵ
        numgrad[i] = (J(vector_to_array(𝚯 + perturb)) - J(vector_to_array(𝚯 - perturb))) / (2*ϵ)
        perturb[i] = 0
    end
    numgrad
end

numGrad = checkGradient(array_to_vector([init_θ1, init_θ2]))
propGrad = array_to_vector(ΔJ([init_θ1, init_θ2]))
@test isapprox(numGrad, propGrad, atol=1e-9)
#

#= show test case
x_test = convert(Array, X[1, :])
@show y_hat = predict(x_test, 𝚯_min)
@show d[1, 1]

function image_digits(x)
    Gray.(reshape(x, 28, 28)')
end

image_digits(x_test)
=#
