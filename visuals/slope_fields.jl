using GLMakie

include("utils.jl")

x_bounds = [-0.5, 2.5]
y_bounds = [-0.5, 2.5]
density = 30

xs = collect(LinRange(x_bounds..., density))
ys = collect(LinRange(y_bounds..., density))

fig = Figure(size=(700, 700))
ax = Axis(fig[1, 1], limits=(x_bounds..., y_bounds...))
set_theme!(merge(theme_light(), theme_latexfonts()))

k = 1
# dy/dx = 
function f(x, y)
    try
        return sqrt(1 - (k^2 * y)) / (k * sqrt(y))
    catch
        return 0
    end
end

function slope_field()
    us_norm = [1/hypot(1, f(x, y)) for x in xs, y in ys]
    vs_norm = [f(x, y)/hypot(1, f(x, y)) for x in xs, y in ys]

    colors = [RGBf(0.5 - 0.5l, 0.5, 0.5) for l in vs_norm]

    arrows!(ax, xs, ys, us_norm, vs_norm, color=vec(colors), arrowsize=0, lengthscale=0.05)

    on(events(fig).mousebutton) do event
        if event.action == Mouse.press
            x_pts = [map_value(events(ax).mouseposition[][1], 50, 684, x_bounds...)]
            y_pts = [map_value(events(ax).mouseposition[][2], 37, 682, y_bounds...)]
            euler_method(f, 0.005, 50, x_pts, y_pts)
            lines!(ax, x_pts, y_pts, color=:black)
        end
    end

    display(fig)
end

slope_field()