using GLMakie

include("../calc_3d.jl")

## - New cell

fig = Figure()
ax = Axis3(fig[1, 1])

function draw()
    xs = collect(LinRange(0, 10, 10))
    ys = collect(LinRange(0, 10, 10))
    zs = [0]
    map_xs = collect(LinRange(0, 10, 100))
    map_ys = collect(LinRange(0, 10, 100))

    points = [Point3f(x, y, z) for x in xs, y in ys, z in zs]

    F(x, y, z) = 0.2*[sin(x), sin(y), 0] - 0.2*[sin(y), sin(x), 0] + [0.5, 0.5, 0]
    # F(x, y, z) = 0.2*[sin(x), sin(y), 0]
    # F(x, y, z) = 0.2*[sin(y), sin(x), 0]
    # F(x, y, z) = 0.2*[sin(x), sin(y), 0] + 0.2*[sin(y), sin(x), 0]

    # Scalar potential
    ϕ(x, y, z) = divergence(F, x, y, z)

    # Vector potential
    A(x, y, z) = curl(F, x, y, z)

    # Curl-free component
    R_cf(x, y, z) = -gradient(ϕ, x, y, z)

    # Divergence-free component
    R_df(x, y, z) = curl(A, x, y, z)

    # Summation of the two components
    R(x, y, z) = R_cf(x, y, z) + R_df(x, y, z)

    # Constant difference between F and R
    Δ(x, y, z) = F(x, y, z) - R(x, y, z)

    # Reconstruction of F
    C(x, y, z) = R(x, y, z) + Δ(x, y, z)

    divF = [ divergence(F, x, y, 0) for x in map_xs, y in map_ys ]
    heatmap!(ax, map_xs, map_ys, divF, colormap=:magma, colorrange=(-1, 1), interpolate=true)

    curlF = [ Vec3f( curl(F, x, y, z) ) for x in xs, y in ys, z in zs ]
    arrows!(ax, vec(points), vec(curlF), lengthscale=1, color=:green, arrowsize=0.05, align=:tail)

    vecsF = [ Vec3f( F(x, y, z) ) for x in xs, y in ys, z in zs ]
    arrows!(ax, vec(points), vec(vecsF), lengthscale=0.2, color=:black, arrowsize=0.05, align=:tail)

    # vecsR_cf = [ Vec3f( R_cf(x, y, z) ) for x in xs, y in ys, z in zs ]
    # arrows!(ax, vec(points), vec(vecsR_cf), lengthscale=0.2, color=:red, arrowsize=0.075, align=:tail)

    # vecsR_df = [ Vec3f( R_df(x, y, z) ) for x in xs, y in ys, z in zs ]
    # arrows!(ax, vec(points), vec(vecsR_df), lengthscale=0.2, color=:red, arrowsize=0.075, align=:tail)

    vecsC = [ Vec3f( C(x, y, z) ) for x in xs, y in ys, z in zs ]
    arrows!(ax, vec(points), vec(vecsC), lengthscale=0.2, color=:red, arrowsize=0.05, align=:tail)

    display(fig)
end

draw()
