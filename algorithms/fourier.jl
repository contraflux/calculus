include("../calc_1d.jl")

using GLMakie

ns = [i for i in 0:10]
ts = [i for i in 0:π/64:4π]

C = [t -> cos(n * t) for n in ns]
S = [t -> sin(n * t) for n in ns]

fig = Figure(size=(600, 600))
ax_t = Axis(fig[1, 1])
ax_f = Axis(fig[2, 1])

function project(f, C, S)
    """ Project a function f onto the frequency basis and return the new
    function. Assume C is a basis of cosine functions and S is a basis of sine
    functions.
    """
    as = [integral(t -> f(t) * c(t), 0, 2π) for c in C] / π
    bs = [integral(t -> f(t) * s(t), 0, 2π) for s in S] / π

    as[1] /= 2 # Correction to normalize the first cosine vector with 2π instead of just π

    scatter!(ax_f, ns, as, color=:orange)
    scatter!(ax_f, ns, bs, color=:blue)
    
    F(x) = sum([(as[n] * C[n](x) + bs[n] * S[n](x)) for n in eachindex(ns)])

    lines!(ax_t, ts, [f(t) for t in ts], color=:black)
    lines!(ax_t, ts, [F(t) for t in ts], color=:red)

    display(fig)

    return F
end

""" Periodic square wave """
# function f(x)
#     if x % 2π < π
#         return 1
#     else
#         return -1
#     end
# end

""" Periodic sawtooth wave """
# function f(x)
#     x = x % 2π
#     return -(x/π) + 1
# end

""" Linear frequency ramp """
# function f(x)
#     x = x % 2π
#     return sin((π/4 * x) * x)
# end

""" Polynomial ceneter around π """
# function f(x)
#     x = x % 2π
#     return (x - π)^3
# end

F = project(f, C, S)
