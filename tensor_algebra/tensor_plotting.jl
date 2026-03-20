using GLMakie
using DifferentialEquations

include("tensors.jl")

function draw_sphere()
    @variables θ φ

    function sphere_points(u, v)
        return [sin(u)cos(v), sin(u)sin(v), cos(u)]
    end

    sphere_basis = [
        Tensor([cos(θ)cos(φ), cos(θ)sin(φ), -sin(θ)]),
        Tensor([-sin(θ)sin(φ), sin(θ)cos(φ), 0])
    ]

    us = range(1e-2, π - 1e-2, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), sphere_basis)
    function geodesic!(du, u, p, t)
        # u = [θ, φ, vθ, vφ]
        # du = derivatives of u
        Γ_num = substitute(Γ, Dict(θ=>u[1], φ=>u[2]))
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -sum([Γ_num[1][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
        du[4] = -sum([Γ_num[2][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
    end

    u0 = [π/2, 0.0, 1.0, -1.0]
    tspan = (0.0, 5.0)
    prob = ODEProblem(geodesic!, u0, tspan)
    sol = solve(prob, abstol=1e-10, reltol=1e-10)

    geodesic_points = [sphere_points(s[1], s[2]) for s in sol.u]
    R_scalar = ricci_scalar((θ, φ), sphere_basis)

    cartesian_points = [sphere_points(u, v) for u in us, v in vs]
    R_scalar_values = [Float64(Symbolics.unwrap(substitute(R_scalar, Dict(θ=>u, φ=>v)))) for u in us, v in vs]

    plot_shape(cartesian_points, R_scalar_values, geodesic_points)
end

function draw_torus(R=3, r=1)
    @variables θ φ

    function torus_points(u, v)
        return [(R + r * cos(u)) * cos(v), (R + r * cos(u)) * sin(v), r * sin(u)]
    end

    torus_basis = [
        Tensor([-r * sin(θ) * cos(φ), -r * sin(θ) * sin(φ), r * cos(θ)]),
        Tensor([-(R + r * cos(θ)) * sin(φ), (R + r * cos(θ)) * cos(φ), 0])
    ]


    us = range(0, 2π, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), torus_basis)
    function geodesic!(du, u, p, t)
        # u = [θ, φ, vθ, vφ]
        # du = derivatives of u
        Γ_num = substitute(Γ, Dict(θ=>u[1], φ=>u[2]))
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -sum([Γ_num[1][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
        du[4] = -sum([Γ_num[2][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
    end

    u0 = [0.0, 0.0, 3.25, -0.5]
    tspan = (0.0, 5.0)
    prob = ODEProblem(geodesic!, u0, tspan)
    sol = solve(prob, abstol=1e-10, reltol=1e-10)

    geodesic_points = [torus_points(s[1], s[2]) for s in sol.u]
    R_scalar = ricci_scalar((θ, φ), torus_basis)

    cartesian_points = [torus_points(u, v) for u in us, v in vs]
    R_scalar_values = [Float64(Symbolics.unwrap(substitute(R_scalar, Dict(θ=>u, φ=>v)))) for u in us, v in vs]

    coarse_us = us[begin:2:end]
    coarse_vs = vs[begin:2:end]
    grid = [torus_points(u, v) for u in coarse_us, v in coarse_vs]
    vecs = [substitute(torus_basis[1], Dict(θ=>u, φ=>v)).data for u in coarse_us, v in coarse_vs]

    plot_shape(cartesian_points, R_scalar_values, geodesic_points, (grid, vecs))
end

function draw_klein(r=3)
    @variables θ φ

    function klein_points(u, v)
        x = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2v)) * cos(u)
        y = (r + cos(u/2)*sin(v) - sin(u/2)*sin(2v)) * sin(u)
        z = sin(u/2)*sin(v) + cos(u/2)*sin(2v)
        return [x, y, z]
    end

    klein_basis = [
        Tensor([
            (-sin(θ/2)/2 * sin(φ) - cos(θ/2)/2 * sin(2φ)) * cos(θ) - (r + cos(θ/2)*sin(φ) - sin(θ/2)*sin(2φ))*sin(θ),
            (-sin(θ/2)/2 * sin(φ) - cos(θ/2)/2 * sin(2φ)) * sin(θ) + (r + cos(θ/2)*sin(φ) - sin(θ/2)*sin(2φ))*cos(θ),
            cos(θ/2)/2 * sin(φ) - sin(θ/2)/2 * sin(2φ)
        ]),
        Tensor([
            (cos(θ/2)*cos(φ) - 2*sin(θ/2)*cos(2φ)) * cos(θ),
            (cos(θ/2)*cos(φ) - 2*sin(θ/2)*cos(2φ)) * sin(θ),
            sin(θ/2)*cos(φ) - 2*cos(θ/2)*cos(2φ)
        ])
    ]

    us = range(0, 2π, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), klein_basis)
    function geodesic!(du, u, p, t)
        # u = [θ, φ, vθ, vφ]
        # du = derivatives of u
        Γ_num = substitute(Γ, Dict(θ=>u[1], φ=>u[2]))
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -sum([Γ_num[1][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
        du[4] = -sum([Γ_num[2][i,j] * u[2+i] * u[2+j] for i in 1:2, j in 1:2])
    end

    u0 = [0.0, 0.0, 3.25, -0.5]
    tspan = (0.0, 5.0)
    prob = ODEProblem(geodesic!, u0, tspan)
    sol = solve(prob, abstol=1e-10, reltol=1e-10)

    geodesic_points = [klein_points(s[1], s[2]) for s in sol.u]
    R_scalar = ricci_scalar((θ, φ), klein_basis)

    cartesian_points = [klein_points(u, v) for u in us, v in vs]
    R_scalar_values = [Float64(Symbolics.unwrap(substitute(R_scalar, Dict(θ=>u, φ=>v)))) for u in us, v in vs]

    plot_shape(cartesian_points, R_scalar_values, geodesic_points)
end

function plot_shape(points, scalar_field, geodesic_points=[], vector_field=())
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    z = [p[3] for p in points]
    color_limit = maximum(abs.(scalar_field))

    # Setup axis
    set_theme!(theme_dark())
    fig = Figure(size=(700, 500), fxaa=true)
    ax = Axis3(fig[1,1], aspect=:data)
    ax.title = "Ricci Scalar Curvature"
    hidespines!(ax)

    # Plot surface
    wireframe!(ax, x, y, z, color=:white, linewidth=0.25, alpha=0.5)
    s = surface!(ax, x, y, z, color=scalar_field, colormap=:RdBu, colorrange=(-color_limit, color_limit))
    Colorbar(fig[1,2], s)

    # Plot geodesic
    if !isempty(geodesic_points)
        xs = [p[1] for p in geodesic_points]
        ys = [p[2] for p in geodesic_points]
        zs = [p[3] for p in geodesic_points]
        lines!(ax, xs, ys, zs, color=:lightblue, linewidth=2)
    end

    # Plot vectors
    if !isempty(vector_field)
        grid = vec([Point3f(x) for x in vector_field[1]])
        vecs = vec([Vec3f(v) for v in vector_field[2]])
        lengths = [norm(v) for v in vector_field[2]]
        arrows!(ax, grid, vecs, color=lengths, colormap=:magma, arrowsize=0.05, lengthscale=0.1)
    end

    display(fig)
end

draw_torus()