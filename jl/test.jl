m = zeros(10,10)

for i in 1:size(m, 1)
    for j in 1:size(m, 2)
        if i == j
            m[i, j] = -2
        end
        if i + 1 == j || j + 1 == i
            m[i,j] = 1
        end
    end
end


function fac(n::T) where T <: BigInt
    if n == 1
        return n
    else
        return fac(n-1) *n
    end
end

@code_warntype fac(100)

n0 = 0
for i in 1:10
    if rand(0:1) == 0
        global n0 += 1
    end
end
@show n0/10



al = [0, 0.5, 0.9]
T = 100
global result = []
for a in al
    tmp = [0.0]
    for i in 2:T+1
        x = a * tmp[i-1] + rand()
        push!(tmp, x)
    end
    push!(result, tmp)
end
using Plots
scatter(result)



#
# mutable struct MyMutableRange
#     a::Int
#     b::Int
#     c
# end
#
# using DataFrames
# df = DataFrame(a = 1:10, b = 10 .* rand(10), c = 10 .* rand(10))
# @df df plot(:a, [:b :c], colour = [:red :blue])
# @df df scatter(:a, :b, markersize = 4 .* log.(:c .+ 0.1))
#
# using RDatasets, StatPlots
# school = RDatasets.dataset("mlmRev","Hsb82")
# @df school density(:MAch, group = :Sx)

using LinearAlgebra, Plots
A = [1 2; 2 1]
evals, evecs = eigen(A)
a1, a2 = evals
eig_1 = [0 0; evecs[:,1]']
eig_2 = [0 0; evecs[:,2]']
x = range(-5, 5, length = 10)
y = -x

plot(eig_1[:, 2], a1 * eig_2[:, 2], arrow = true, color = :red,
legend = :none, xlims = (-3, 3), ylims = (-3, 3), xticks = -3:3, yticks = -3:3, framestyle = :origin)
# plot!(a2 * eig_1[:, 2], a2 * eig_2, arrow = true, color = :red) plot!(eig_1, eig_2, arrow = true, color = :blue)
# plot!(x, y, color = :blue, lw = 0.4, alpha = 0.6)
# plot!(x, x, color = :blue, lw = 0.4, alpha = 0.6)
