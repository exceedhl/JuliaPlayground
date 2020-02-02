using Plots, Images, CSV
using LineSearches, Optim

include("nn.jl")
include("util.jl")

d = CSV.read("./digits/train.csv")
X = convert(Array, d[!, 2:end] ./ 255)
Y = map(x->transform_digit_to_bitvec(x, 10), d[!, 1])

m = 40000
X_train = X[1:m, :]
Y_train = Y[1:m]

L1 = 784
L2 = 50
L3 = 10
λ = 1

ϵ = 0.12
init_θ = rand(Distributions.Uniform(-ϵ, ϵ), (L1 + 1) * L2 + (L2 + 1) * L3)

# use Optim to find minimum 𝚯
function f(θ_vec)
    J(vector_to_array(θ_vec))
end

function g!(storage, θ_vec)
    g = array_to_vector(ΔJ(vector_to_array(θ_vec)))
    for i = 1:length(storage)
        storage[i] = g[i]
    end
end

@time @show result = Optim.optimize(
    f,
    g!,
    init_θ,
    LBFGS(m=5, alphaguess=LineSearches.InitialHagerZhang(), linesearch=LineSearches.MoreThuente()),
    # LBFGS(m=20, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe()),
    Optim.Options(iterations = 5, show_trace=true, show_every=1)
    # ConjugateGradient()
    # ConjugateGradient(eta=0.1, alphaguess=LineSearches.InitialQuadratic(), linesearch = LineSearches.StrongWolfe())
)
𝚯_min = vector_to_array(result.minimizer)

X_test = X[40001:42000, :]
Y_test = Y[40001:42000]
y_pred = [transform_bitvec_to_digit(predict(X_test[i, :], 𝚯_min)) for i in 1:length(Y_test)]
@show mean(y_pred .!= d[40001:42000, 1])

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

#= show test case
x_test = convert(Array, X[1, :])
@show y_hat = predict(x_test, 𝚯_min)
@show d[1, 1]

function image_digits(x)
    Gray.(reshape(x, 28, 28)')
end

image_digits(x_test)
=#
