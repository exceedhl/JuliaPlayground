using MLJ

@load UnivariateStandardizer pkg=MLJModels

v = Array{Float64, 1}(1:10)
m = machine(UnivariateStandardizer(), v)
fit!(m)
vv = transform(m, v)

using StatsBase
mean(vv)
mean(v)
std(v)
std(vv)
