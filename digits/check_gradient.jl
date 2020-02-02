using Distributions, Test

include("nn.jl")
include("util.jl")

L1 = 3
L2 = 3
L3 = 3
λ = 1

m = 5
X_train = reshape(rand(m * L1), m, L1)
Y_train = map(x -> transform_digit_to_bitvec(x, 3), rand(0:2, m))

nn = BasicNN([L1, L2, L3], λ)

function checkGradient(𝚯)
    ϵ = 1e-4
    𝚯 = array_to_vector(𝚯)
    numgrad = zeros(size(𝚯))
    perturb = zeros(size(𝚯))
    for i = 1:length(𝚯)
        perturb[i] = ϵ
        nn.𝚯 = vector_to_array(𝚯 + perturb, nn.L)
        J2 = J(nn, X_train, Y_train)
        nn.𝚯 = vector_to_array(𝚯 - perturb, nn.L)
        J1 = J(nn, X_train, Y_train)
        numgrad[i] = (J2 - J1) / (2 * ϵ)
        perturb[i] = 0
    end
    nn.𝚯 = vector_to_array(𝚯, nn.L)
    numgrad
end

numGrad = checkGradient(nn.𝚯)
propGrad = array_to_vector(ΔJ(nn, X_train, Y_train))
@test isapprox(numGrad, propGrad, atol = 1e-9)
