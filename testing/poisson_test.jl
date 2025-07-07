using GLMakie
# using FFTW
# using Interpolations

include("../calc_3d.jl")

## - New cell

fig = Figure()
ax = Axis(fig[1, 1], limits=(-10, 10, -10, 10))

mutable struct pointCharge
    x::Vector{Float64}
    v::Vector{Float64}
    q::Float64
    m::Float64
end

xs = collect(LinRange(-10, 10, 25))
ys = collect(LinRange(-10, 10, 25))

Δx = xs[2] - xs[1]
Δy = ys[2] - ys[1]

points = [Point2f(x, y) for x in xs, y in ys]

pc1 = pointCharge([5, 0.0], [-1, 0.0], 1.0, 1.0)
pc2 = pointCharge([0.0, -5], [0.0, 1], -1.0, 1.0)
point_charges = [pc1, pc2]

function charge_density(x, y)
    ρ = 0
    for pc in point_charges
        x_bounds = Δx - abs(pc.x[1] - x)
        y_bounds = Δy - abs(pc.x[2] - y)
        if x_bounds > 0 && y_bounds > 0
            m = (x_bounds * y_bounds) / (Δx * Δy)
            ρ += pc.q * m
        end
    end
    return ρ
end

function get_potential(xs, ys, ρ, x, y)
    ϕ = 0
    for x_p in xs
        for y_p in ys
            if x != x_p || y != y_p
                ϕ += ρ(x_p, y_p) / hypot(x - x_p, y - y_p)
            end
        end
    end
    for pc in point_charges
        if x == pc.x[1] && y == pc.x[2]
            return Inf
        end
    end
    return ϕ
end


function draw()
    """Goal: Create a scalar field ϕ such that F = -∇ϕ, where ∇⋅F = ρ. 
        Put differently, we want to find a scalar field ϕ such that ∇⋅(∇ϕ) = ρ"""

    #TODO: Make a poisson solver please lol

    ρ(x, y) = charge_density(x, y)

    ρ_grid = [ ρ(x, y) for x in xs, y in ys ]

    heatmap!(ax, xs, ys, ρ_grid, colormap = :plasma)

    ϕ(x, y) = get_potential(xs, ys, ρ, x, y)

    ϕ_grid = [ ϕ(x, y) for x in xs, y in ys ]
    heatmap!(ax, xs, ys, ϕ_grid, colormap = :plasma, interpolate=true)

    E(x, y) = -gradient(ϕ, x, y)

    E_grid = [ Vec2f( E(x, y) ) for x in xs, y in ys]
    arrows!(ax, vec(points), vec(E_grid), color=:black, lengthscale=0.5)

    display(fig)
end

draw()
