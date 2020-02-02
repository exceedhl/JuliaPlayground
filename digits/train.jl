using Plots, Images, CSV

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

X_test = X[40001:42000, :]
Y_test = Y[40001:42000]

nn = BasicNN([L1, L2, L3], λ)
train!(nn, X_train, Y_train, iterations = 1)
y_pred = [transform_bitvec_to_digit(predict(nn, X_test[i, :])) for i in 1:size(X_test, 1)]
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
