include("../calc_1d.jl")

using GLMakie

f(x) = x^2

function newton_method(f, x, iters, plot)
    """ An algorithm to find zeros of a function f by repeatedly finding the x-intercepts of a tangent lines from an initial x value
    Parameters: f:R -> R, x ∈ R, iters ∈ Z
    """
    fig = Figure()
    ax = Axis(fig[1, 1], limits=(-5, 5, -5, 5))
    lines!(ax, [x for x in -10:0.1:10], [f(x) for x in -10:0.1:10])
    for _ in 1:iters
        y = f(x)
        lines!(ax, [x, x], [y, 0], color=:black)
        m = derivative(f, x)
        scatter!(ax, [x], [y], color=:black)
        lines!(ax, [x, x-(y/m)], [y, 0], color=:green)
        x = x - (y / m)
        if plot
            display(fig)
            sleep(0.5)
        end
    end
    return x
end
