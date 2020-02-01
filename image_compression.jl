using Images, TestImages
using MLJ
using FileIO

img = FileIO.load("bird_small.png")
# img = testimage("mandrill")
data = reshape(channelview(img), 3, length(img))

@load KMeans
X = coerce(data, Continuous)'
m = machine(KMeans(k=16), X)
fit!(m)

assignments = report(m).assignments
centers = fitted_params(m).centers

for i = 1:size(X, 1)
    X[i, :] = centers[:, assignments[i]]
end

output_img_raw = reshape(X', 3, size(img)...)

using Plots
l = @layout [a{0.5w};b{0.5h}]
p1 = plot(img, size=size(img), framestyle=:none)
p2 = plot(colorview(RGB, output_img_raw), size=size(img), framestyle=:none)
plot(p1, p2, layout=l, size=(size(img).*2))
