using GLMakie
using DifferentialEquations

include("tensors.jl")

function get_sphere(θ, φ)
    function points(u, v)
        return [sin(u)cos(v), sin(u)sin(v), cos(u)]
    end

    basis = [
        Tensor([cos(θ)cos(φ), cos(θ)sin(φ), -sin(θ)]),
        Tensor([-sin(θ)sin(φ), sin(θ)cos(φ), 0])
    ]
    
    us = range(1e-2, π - 1e-2, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), basis)
    ∂ = PartialDerivative((θ, φ))
    ∇ = CovariantDerivative(Γ, ∂)
    R_scalar = ricci_scalar((θ, φ), basis)

    return points, basis, (us, vs), Γ, ∂, ∇, R_scalar
end

function get_torus(θ, φ, R=3, r=1)
    function points(u, v)
        return [(R + r * cos(u)) * cos(v), (R + r * cos(u)) * sin(v), r * sin(u)]
    end

    basis = [
        Tensor([-r * sin(θ) * cos(φ), -r * sin(θ) * sin(φ), r * cos(θ)]),
        Tensor([-(R + r * cos(θ)) * sin(φ), (R + r * cos(θ)) * cos(φ), 0])
    ]

    us = range(0, 2π, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), basis)
    ∂ = PartialDerivative((θ, φ))
    ∇ = CovariantDerivative(Γ, ∂)
    R_scalar = ricci_scalar((θ, φ), basis)

    return points, basis, (us, vs), Γ, ∂, ∇, R_scalar
end

function get_klein(θ, φ, r=3)
    function points(u, v)
        x = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2v)) * cos(u)
        y = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2v)) * sin(u)
        z = sin(u/2)*sin(v) + cos(u/2)*sin(2v)
        return [x, y, z]
    end

    basis = [
        Tensor([
            (-sin(θ/2)/2 * sin(φ) - cos(θ/2)/2 * sin(2φ)) * cos(θ) - (r + cos(θ/2)*sin(φ) - sin(θ/2)*sin(2φ))*sin(θ),
            (-sin(θ/2)/2 * sin(φ) - cos(θ/2)/2 * sin(2φ)) * sin(θ) + (r + cos(θ/2)*sin(φ) - sin(θ/2)*sin(2φ))*cos(θ),
            cos(θ/2)/2 * sin(φ) - sin(θ/2)/2 * sin(2φ)
        ]),
        Tensor([
            (cos(θ/2)*cos(φ) - 2*sin(θ/2)*cos(2φ)) * cos(θ),
            (cos(θ/2)*cos(φ) - 2*sin(θ/2)*cos(2φ)) * sin(θ),
            sin(θ/2)*cos(φ) + 2*cos(θ/2)*cos(2φ)
        ])
    ]

    us = range(0, 2π, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), basis)
    ∂ = PartialDerivative((θ, φ))
    ∇ = CovariantDerivative(Γ, ∂)
    R_scalar = ricci_scalar((θ, φ), basis)

    return points, basis, (us, vs), Γ, ∂, ∇, R_scalar
end

function geodesic!(Γ, du, u, p, t)
    # u = [θ, φ, vθ, vφ]
    # du = derivatives of u
    Γ_num = substitute(Γ, Dict(θ=>u[1], φ=>u[2]))
    du[1] = u[3]
    du[2] = u[4]
    du[3] = -sum([Γ_num[1][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
    du[4] = -sum([Γ_num[2][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
end

function plot_surface!(ax, points, values=nothing)
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    z = [p[3] for p in points]
    if isnothing(values)
        wireframe!(ax, x, y, z, color=:white, linewidth=0.25, alpha=0.5)
        s = surface!(ax, x, y, z, colormap=[:grey, :grey])
        return s
    end
    color_limit = maximum(abs.(values))
    wireframe!(ax, x, y, z, color=:white, linewidth=0.25, alpha=0.5)
    s = surface!(ax, x, y, z, color=values, colormap=:RdBu, colorrange=(-color_limit, color_limit))
    return s
end

function plot_line!(ax, points)
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    zs = [p[3] for p in points]
    lines!(ax, xs, ys, zs, color=:lightblue, linewidth=2)
end

function plot_geodesic!(ax, points, Γ, u0, tspan)
    problem = ODEProblem((du, u, p, t) -> geodesic!(Γ, du, u, p, t), u0, tspan)
    solution = solve(problem, abstol=1e-10, reltol=1e-10)
    geodesic_points = [points(u_t[1], u_t[2]) for u_t in solution.u]
    plot_line!(ax, geodesic_points)
end

function plot_vectors!(ax, grid, vecs; lengthscale=1, arrowscale=1, colormap=:viridis, normalize=false)
    grid = vec([Point3f(x) for x in grid])
    vecs = vec([Vec3f(v) for v in vecs])
    lengths = vec([norm(v) for v in vecs])
    arrows!(ax, grid, vecs, color=lengths, colormap=colormap, arrowsize=arrowscale, lengthscale=lengthscale, normalize=normalize)
end

@variables θ φ

points, basis, (us, vs), Γ, ∂, ∇, R_scalar = get_klein(θ, φ)

# Surface points
cartesian_points = [points(u, v) for u in us, v in vs]
scalar_field = [Float64(Symbolics.unwrap(substitute(R_scalar, Dict(θ=>u, φ=>v)))) for u in us, v in vs]

# Geodesic conditions
u0 = [π/2, 0.0, 1.0, -1.0]
tspan = (0.0, 5.0)

# Vector field
coarse_us = us[begin:2:end]
coarse_vs = vs[begin:2:end]
grid = [points(u, v) for u in coarse_us, v in coarse_vs]
X = Tensor([0, sin(φ)])
div_X = ∇[:i] * X[:i]
div_X_values = [Float64(Symbolics.unwrap(substitute(div_X, Dict(θ=>u, φ=>v)))) for u in us, v in vs]
vecs = [
    substitute((
        X.data[1] * basis[1][:i] + 
        X.data[2] * basis[2][:i]
    ).tensor, Dict(θ=>u, φ=>v)).data
    for u in coarse_us, v in coarse_vs
]

set_theme!(theme_dark())
fig = Figure(size=(700, 500), fxaa=true)
ax = Axis3(fig[1,1], aspect=:data)
ax.title = "Divergence Plot"
hidespines!(ax)

s = plot_surface!(ax, cartesian_points, div_X_values)
Colorbar(fig[1,2], s)
# plot_geodesic!(ax, points, Γ, u0, tspan)
plot_vectors!(ax, grid, vecs, lengthscale=0.05, arrowscale=0.015, colormap=:magma, normalize=true)

display(fig)