using Distributions, Test

include("nn.jl")
include("util.jl")

L1 = 3
L2 = 3
L3 = 3
λ = 1
m = 5
ϵ = 0.12

X_train = reshape(rand(m * L1), m, L1)
Y_train = map(x -> transform_digit_to_bitvec(x, 3), rand(0:2, m))

init_θ = rand(Distributions.Uniform(-ϵ, ϵ), (L1 + 1) * L2 + (L2 + 1) * L3)

function checkGradient(𝚯)
    ϵ = 1e-4
    numgrad = zeros(size(𝚯))
    perturb = zeros(size(𝚯))
    for i = 1:length(𝚯)
        perturb[i] = ϵ
        numgrad[i] = (J(vector_to_array(𝚯 + perturb)) -
                      J(vector_to_array(𝚯 - perturb))) / (2 * ϵ)
        perturb[i] = 0
    end
    numgrad
end

numGrad = checkGradient(init_θ)
propGrad = array_to_vector(ΔJ(vector_to_array(init_θ)))
@test isapprox(numGrad, propGrad, atol = 1e-9)
