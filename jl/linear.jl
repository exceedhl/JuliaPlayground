using Plots

x_vals = [0 0 0 ; 2 -3 -4]
y_vals = [0 0 0 ; 4 3 -3.5]
plot(x_vals, y_vals, arrow = true, color = :green, legend = :none, xlims = (-5, 5), ylims = (-5, 5), annotations = [(2.2, 4.4, "[2, 4]"),
                   (-3.3, 3.3, "[-3, 3]"),
                   (-4.4, -3.85, "[-4, -3.5]")],
    xticks = -5:1:5, yticks = -5:1:5,
    framestyle = :origin)

using LinearAlgebra
dot(x_vals, y_vals)


# fixed linear function, to generate a plane
f(x, y) = 0.2x + 0.1y
# lines to vectors
x_vec = [0 0; 3 3]
y_vec = [0 0; 4 -4]
z_vec = [0 0; f(3, 4) f(3, -4)]
# draw the plane
n = 20
grid = range(-5, 5, length = n)
z2 = [ f(grid[row], grid[col]) for row in 1:n, col in 1:n ]
wireframe(grid, grid, z2, fill = :blues, gridalpha =1 )
plot!(x_vec, y_vec, z_vec, color = [:blue :green], linewidth = 3, labels = "",
colorbar = false)
