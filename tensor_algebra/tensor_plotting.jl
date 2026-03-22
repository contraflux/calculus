using Makie
using GLMakie
using DifferentialEquations

include("tensors.jl")

function get_sphere(θ, φ)
    function points(u, v)
        return [sin(u)cos(v), sin(u)sin(v), cos(u)]
    end

    basis = Basis([
        Tensor([cos(θ)cos(φ), cos(θ)sin(φ), -sin(θ)]),
        Tensor([-sin(θ)sin(φ), sin(θ)cos(φ), 0])
    ])
    
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

    basis = Basis([
        Tensor([-r * sin(θ) * cos(φ), -r * sin(θ) * sin(φ), r * cos(θ)]),
        Tensor([-(R + r * cos(θ)) * sin(φ), (R + r * cos(θ)) * cos(φ), 0])
    ])

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

    basis = Basis([
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
    ])

    us = range(0, 2π, 50)
    vs = range(0, 2π, 50)

    Γ = christoffel((θ, φ), basis)
    ∂ = PartialDerivative((θ, φ))
    ∇ = CovariantDerivative(Γ, ∂)
    R_scalar = ricci_scalar((θ, φ), basis)

    return points, basis, (us, vs), Γ, ∂, ∇, R_scalar
end

function get_normal(basis, u, v)
    e1 = evaluate(basis[1], Dict(θ=>u, φ=>v)).data
    e2 = evaluate(basis[2], Dict(θ=>u, φ=>v)).data
    n = cross(e1, e2)
    return n / norm(n)
end

function geodesic!(Γ, du, u, p, t)
    # u = [θ, φ, vθ, vφ]
    # du = derivatives of u
    Γ_num = evaluate(Γ, Dict(θ=>u[1], φ=>u[2]))
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

function plot_geodesic!(ax, points, normal, Γ, u0, tspan; offset=0.01)
    problem = ODEProblem((du, u, p, t) -> geodesic!(Γ, du, u, p, t), u0, tspan)
    solution = solve(problem, abstol=1e-10, reltol=1e-10)
    geodesic_points = [points(u_t[1], u_t[2]) + offset * normal(u_t[1], u_t[2]) for u_t in solution.u]
    plot_line!(ax, geodesic_points)
end

function plot_vectors!(ax, grid, vecs; lengthscale=1, arrowscale=0.1, colormap=:viridis, normalize=false)
    grid = vec([Point3f(x) for x in grid])
    vecs = vec([Vec3f(v) for v in vecs])
    lengths = vec([norm(v) for v in vecs])
    if normalize
        vecs = [v / norm(v) for v in vecs]
    end
    arrows3d!(ax, grid, vecs, color=lengths, colormap=colormap,
        lengthscale=lengthscale, markerscale=arrowscale,
        align=:tail
    )
end

function plot_form!(ax, points, ω, us, vs; inward=false, tspan=10.0, n_seeds=10, colormap=:viridis, linewidth=1, abstol=1e-10, reltol=1e-10, offset=0.01, closure_tol=0.05, coverage_tol=0.1)
    
    curves = []
    mags = []

    function wrap(u, v)
        u_out = mod(u - us[begin], us[end] - us[begin]) + us[begin]
        v_out = mod(v - vs[begin], vs[end] - vs[begin]) + vs[begin]
        return u_out, v_out
    end

    function already_covered(u0, v0)
        p = points(u0, v0)
        for curve in curves
            for pt in curve
                if pt !== nothing && norm(pt - p) < coverage_tol
                    return true
                end
            end
        end
        return false
    end

    function integrate_curve(u0, v0, mag)
        p_start = points(wrap(u0, v0)...)
        prob = ODEProblem(
            (du, u, p, t) -> begin
                wu, wv = wrap(u[1], u[2])
                ω_n = evaluate(ω, Dict(θ=>wu, φ=>wv))
                mag_n = norm(ω_n.data)
                if mag_n < 1e-10
                    du[1] = 0.0
                    du[2] = 0.0
                else
                    du[1] = -ω_n.data[2] / mag_n^2
                    du[2] = ω_n.data[1] / mag_n^2
                end
            end,
            [u0, v0], (0.0, tspan)
        )

        closure_condition(u, t, integrator) = begin
            wu, wv = wrap(u[1], u[2])
            p_current = points(wu, wv)
            t > 0.1 ? norm(p_current - p_start) - closure_tol : closure_tol
        end
        closure_affect!(integrator) = terminate!(integrator)
        cb = ContinuousCallback(closure_condition, closure_affect!)

        sol = solve(prob, abstol=abstol, reltol=reltol, callback=cb, saveat=0.01)
        curve = []
        curve_mags = Float64[]
        prev_pt = nothing
        for u_t in sol.u
            wu, wv = wrap(u_t[1], u_t[2])
            k = inward ? -1 : 1
            pt = points(wu, wv) + offset * k * get_normal(basis, wu, wv)
            if prev_pt !== nothing && norm(pt - prev_pt) > 0.5
                push!(curve, nothing)
                push!(curve_mags, NaN)
            end
            push!(curve, pt)
            ω_n = evaluate(ω, Dict(θ=>wu, φ=>wv))
            push!(curve_mags, norm(ω_n.data))
            prev_pt = pt
        end
        push!(curves, curve)
        push!(mags, curve_mags)
    end

    u_start = (us[begin] + us[end]) / 2
    v_start = (vs[begin] + vs[end]) / 2
    u_step = (us[end] - us[begin]) / n_seeds
    v_step = (vs[end] - vs[begin]) / n_seeds

    for sign in [1.0, -1.0]
        u0, v0 = u_start, v_start
        for i in 1:n_seeds
            ω_num = evaluate(ω, Dict(θ=>u0, φ=>v0))
            mag = norm(ω_num.data)
            if mag < 1e-10
                break
            end

            if !already_covered(u0, v0)
                integrate_curve(u0, v0, mag)
            end

            ω1, ω2 = ω_num.data[1], ω_num.data[2]
            step_size = sqrt(u_step^2 + v_step^2)
            u0, v0 = wrap(u0 + sign * step_size * ω1 / mag, v0 + sign * step_size * ω2 / mag)
        end
    end

    all_mags = filter(isfinite, vcat(mags...))
    colorrange = (minimum(all_mags), maximum(all_mags))
    for (curve, curve_mags) in zip(curves, mags)
        segments = []
        seg_mags = []
        current_seg = []
        current_mags = Float64[]
        for (pt, m) in zip(curve, curve_mags)
            if pt === nothing
                if length(current_seg) > 1
                    push!(segments, current_seg)
                    push!(seg_mags, current_mags)
                end
                current_seg = []
                current_mags = Float64[]
            else
                push!(current_seg, pt)
                push!(current_mags, m)
            end
        end
        if length(current_seg) > 1
            push!(segments, current_seg)
            push!(seg_mags, current_mags)
        end
        for (seg, sm) in zip(segments, seg_mags)
            xs = [p[1] for p in seg]
            ys = [p[2] for p in seg]
            zs = [p[3] for p in seg]
            lines!(ax, xs, ys, zs, color=sm, colormap=colormap, colorrange=colorrange, linewidth=linewidth)
        end
    end
end

@variables θ φ

points, basis, (us, vs), Γ, ∂, ∇, R_scalar = get_torus(θ, φ)

# Surface points
cartesian_points = [points(u, v) for u in us, v in vs]
scalar_field = [evaluate(R_scalar, Dict(θ=>u, φ=>v)) for u in us, v in vs]

# Geodesic conditions
u0 = [π/2, 0.0, 1.0, -1.0]
tspan = (0.0, 5.0)

# Vector field
coarse_us = us[begin:2:end]
coarse_vs = vs[begin:2:end]
grid = [points(u, v) for u in coarse_us, v in coarse_vs]
X = Tensor([1, sin(φ)])
div_X = ∇[:i] * X[:i]
div_X_values = [evaluate(div_X, Dict(θ=>u, φ=>v)) for u in us, v in vs]
vecs = [
    evaluate(X[:i] * basis[:i], Dict(θ=>u, φ=>v)).data
    for u in coarse_us, v in coarse_vs
]
ω = Tensor([1, sin(φ)]')

set_theme!(theme_dark())
fig = Figure(size=(700, 500), fxaa=true)
ax = Axis3(fig[1,1], aspect=:data)
ax.title = "Covector Field"
hidespines!(ax)

s = plot_surface!(ax, cartesian_points)
# Colorbar(fig[1,2], s)
# plot_geodesic!(ax, points, (u, v) -> get_normal(basis, u, v), Γ, u0, tspan)
plot_vectors!(ax, grid, vecs, 
    lengthscale=0.1, arrowscale=0.15, colormap=:magma, normalize=false
)
plot_form!(ax, points, ω, us, vs, 
    inward=true, colormap=:magma,
    tspan=20.0, n_seeds=20, closure_tol=5e-3, coverage_tol=0.15, abstol=1e-12, reltol=1e-12
)

display(fig)