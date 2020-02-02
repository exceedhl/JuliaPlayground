sigmoid(z) = 1.0 ./ (1.0 + exp(-z))

function H(x, 𝚯)
    θ1, θ2 = 𝚯
    a1 = x
    a2 = sigmoid.(θ1 * [1; x])
    a3 = sigmoid.(θ2 * [1; a2])
    [a1, a2, a3]
end

function cost(a, y)
    -log.(a[end])' * y - log.(1 .- a[end])' * (1 .- y)
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
    H(x,𝚯)[end]
end
