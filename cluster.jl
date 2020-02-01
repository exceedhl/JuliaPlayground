using MLJ, Random, CSV
using Plots

function mean_square_distance(x1, x2)
    sum((x1-x2).^2)
end

function loss_of_kmeans(data, assignments, centers)
    sum = 0
    for d in data, i in assignments
        sum += mean_square_distance(d, centers[:, i])
    end
    sum / length(assignments)
end

d = CSV.read("ex7data1.csv")
@load KMeans

ks = 2:2
losses = []
for k in ks
    km2 = KMeans(k=k)
    # @load PCA pkg=MultivariateStats
    # @pipeline SPCA2(std = Standardizer(),
    #                 pca = PCA(),
    #                 km = KMeans(k=3))
    #
    # spca2_mdl = SPCA2()
    spca2 = machine(km2, d)
    fit!(spca2)

    assignments = report(spca2).assignments
    centers = fitted_params(spca2).centers
    loss = loss_of_kmeans(eachrow(convert(Array, d)), assignments, centers)
    @show loss
    push!(losses, loss)
    scatter(d[1], d[2], group=assignments)
    scatter!(centers[1, :], centers[2, :], shape=:cross, ms=8, msw=0.8)
end
# plot(ks, losses, xlabel="number of clusters", ylabel="loss", label="loss")
