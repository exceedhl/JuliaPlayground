using MLJ, RDatasets, PrettyPrinting, Random

# Random.seed!(3203)
X0 = randn(20, 2)
y = vcat(-ones(10), ones(10))

using Plots

ym1 = y .== -1
ym2 = .!ym1
plot(X0[ym1, 1], X0[ym1, 2], seriestype=:scatter, shape=:circle, color=:red)
plot!(X0[ym2, 1], X0[ym2, 2], seriestype=:scatter, shape=:cross, color=:blue)

X = MLJ.table(X0)
y = categorical(y)

@load SVC pkg=LIBSVM

svc_mdl = SVC()
svc = machine(svc_mdl, X, y)

fit!(svc);

ypred = predict(svc, X)
misclassification_rate(ypred, y)

rc = range(svc_mdl, :cost, lower=0.1, upper=15)
tm = TunedModel(model=svc_mdl, ranges=[rc], tuning=Grid(resolution=10),
                resampling=CV(nfolds=3, rng=33), measure=misclassification_rate)
mtm = machine(tm, X, y)

fit!(mtm)

ypred = predict(mtm, X)
misclassification_rate(ypred, y)

r=report(mtm)

# plot SVN contour
XX1 = range(-2, 2, length=100)
XX2 = range(-2, 2, length=100)
YY = zeros(100, 100)
for i in 1:100
    for j in 1:100
        YY[i, j] = convert(Float64, predict(mtm, hcat(XX1[i],XX2[j]))[1])
    end
end
plot!(contour!(XX1, XX2, (x,y) -> convert(Float64, predict(mtm, hcat(x,y))[1]), levels=1, fill=false))

# x = 1:0.5:20
# y = 1:0.5:10
# f(x, y) = begin
#         (3x + y ^ 2) * abs(sin(x) + cos(y))
#     end
# X = repeat(reshape(x, 1, :), length(y), 1)
# Y = repeat(y, 1, length(x))
# Z = map(f, X, Y)
# p1 = contour(x, y, f, fill=true)
# p2 = contour(x, y, Z)
# plot(p1, p2)

# x = y = range(-20, 20, length = 100)
# plot(contour(x, y, (x, y) -> x^2 + y^2))
