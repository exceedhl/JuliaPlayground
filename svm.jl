using MLJ, CSV, ScientificTypes
using Plots

d = CSV.read("ex6data3.csv")
y, X = unpack(d, ==(:y), !=(:y); :y => OrderedFactor)
train, test = partition(1:length(y), 0.8, shuffle = false)

@load SVC pkg = LIBSVM
svc_mdl = SVC()
rc = range(svc_mdl, :cost, lower = 0.1, upper = 100)
tm = TunedModel(
    model = svc_mdl,
    ranges = [rc],
    tuning = Grid(resolution = 100),
    resampling = CV(nfolds = 3),
    measure = f1,
)
svc = machine(tm, X, y)
fit!(svc, rows = train, verbosity=2, force=true)
ypred = predict(svc, rows = test)
f1(ypred, y[test])

# draw hyper-param vs cost line
r = report(svc)
plot(
    r.parameter_values[:, 1],
    r.measurements,
    xlabel = r.parameter_names,
    ylabel = "f1",
    seriestype = :line,
)

# draw data points
scatter(X.x1[test], X.x2[test], group = y[test])
# draw SVM decision boundary
XX1 = range(-0.5, 0.5, length = 100)
XX2 = range(-0.5, 0.5, length = 100)
plot!(contour!(
    XX1,
    XX2,
    (x, y) -> convert(Float64, predict(svc, [x y])[1]),
    levels = 1,
    fill = false,
))

evaluate!()
