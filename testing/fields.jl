include("../calc_nd.jl")

using GLMakie

## 

mutable struct pointCharge
    x::Function
    q::Float64
end

# const ϵ₀ = 8.854187817e-12
const ϵ₀ = 1.0
const μ₀ = 1.256637061e-6
const c = 1.0

function A(r, t)
end

function ϕ(r, t)
    ϕ = 0
    for charge in charges
        radius = hypot((r - charge.x(t))...)
        ϕ += charge.q / (4π * ϵ₀ * radius)
    end
    return ϕ
end

function E(r, t)
    E = -gradient((x, y, z) -> ϕ([x, y, z], t), r...) - partial_t(curl((x, y, z) -> A([x, y, z], t), r...), r, t)
end

function B(r, t)
    B = curl((x, y, z) -> A([x, y, z], t), r...)
end

charge1 = pointCharge(t -> [0, t, 0], 1)
charge2 = pointCharge(t -> [cos(t), sin(t), 0], -1)
charges = [charge1, charge2]

Δt = 0.1

fig = Figure()
ax = Axis3(fig[1, 1])

grid = collect(LinRange(-10, 10, 20))
points = [Point3f(x, y, z) for x in grid, y in grid, z in grid]

for t in 0:Δt:Δt*100
    empty!(ax)

    E_grid = [ Vec3f( -gradient((x, y, z) -> ϕ([x, y, z], t), x, y, z) ) for x in grid, y in grid, z in grid]
    arrows!(ax, vec(points), vec(E_grid), arrowsize=0.01)

    display(fig); sleep(0.01)
end