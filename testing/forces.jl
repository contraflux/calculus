using GLMakie
using Interpolations

include("../calc_3d.jl")

## - Constants

fig = Figure()
ax = Axis3(fig[1, 1])

xs = collect(LinRange(-10, 10, 10))
ys = collect(LinRange(-10, 10, 10))
zs = collect(LinRange(-10, 10, 10))

Δx = xs[2] - xs[1]
Δy = ys[2] - ys[1]
Δz = zs[2] - zs[1]

points = [ Point3f(x, y, z) for x in xs, y in ys, z in zs ]

dt = 0.1

mutable struct pointCharge
    x::Vector{Float64}
    v::Vector{Float64}
    q::Float64
    m::Float64
end

## - Actual code

function update_charges()
    for pc in point_charges
        pc.x += pc.v * dt
    end
end

pc1 = pointCharge([5, 0.0, 0.0], [-1, 0.0, 0.0], 1.0, 1.0)
pc2 = pointCharge([0.0, -5, 0.0], [0.0, 1, 0.0], -1.0, 1.0)
point_charges = [pc1, pc2]

function ϕ(x, y, z)
    ϕ = 0
    for charge in point_charges
        r = hypot(charge.x[1] - x, charge.x[2] - y, charge.x[3] - z)
        ϕ += charge.q / (4 * π * r)
    end
    return ϕ
end

function J(x, y, z)
    J = zeros(3)
    for pc in point_charges
        x_bounds = Δx - abs(pc.x[1] - x)
        y_bounds = Δy - abs(pc.x[2] - y)
        z_bounds = Δz - abs(pc.x[3] - z)
        if x_bounds > 0 && y_bounds > 0 && z_bounds > 0
            m = (x_bounds * y_bounds * z_bounds) / (Δx * Δy * Δz)
            J += pc.q * pc.v * m
        end
    end
    return J
end

E_accumulation = [zeros(3) for x in xs, y in ys, z in zs]
B_accumulation = [zeros(3) for x in xs, y in ys, z in zs]

B_initial = [zeros(3) for x in xs, y in ys, z in zs]

function grid_to_function(phi_grid::Array{Vector{Float64},3}, dx::Float64=1.0)
    nx, ny, nz = size(phi_grid)
    
    # Define axis coordinates
    x = collect(0:dx:(nx - 1) * dx)
    y = collect(0:dx:(ny - 1) * dx)
    z = collect(0:dx:(nz - 1) * dx)

    # Create the interpolant
    itp = interpolate((x, y, z), phi_grid, Gridded(Linear()))
    ext = extrapolate(itp, Line())  # or use Throw(), Line(), or value

    # Return callable function
    return (x, y, z) -> ext(x, y, z)
end

μ₀ = 1
ϵ₀ = 1

for t in 0:dt:dt*100
    empty!(ax)

    B = grid_to_function(B_initial, Δx)


    E_grid = [ Vec3f( -gradient(ϕ, x, y, z) ) for x in xs, y in ys, z in zs]
    E_accumulation += [ ( curl(B, x, y, z) / μ₀ - J(x, y, z) ) / ϵ₀ for x in xs, y in ys, z in zs ] * dt
    E_grid += E_accumulation

    E = grid_to_function(E_grid, Δx)
    arrows!(ax, vec(points), vec(E_grid), lengthscale=15, arrowsize=0.025)

    # ∇×E = -∂B/∂t
    # B += -(∇×E) * ∂t

    B_accumulation += [ curl(E_grid, x, y, z) * dt for x in xs, y in ys, z in zs ]

    # ∇×B = μ₀(J + ϵ₀∂E/∂t)
    # E += (((∇×B)/μ₀ - J)/ϵ₀) * ∂t
    
    update_charges()

    display(fig)
    sleep(0.001)
end