include("calc_1d.jl")
include("calc_2d.jl")

function partial_x(f, x, y, z)
    """ Evaluate the partial derivative of f with respect to x at (x, y, z) 
    Parameters: f:R³ -> R, (x, y, z) ∈ R³
    Returns: ∂f/∂x
    """
    try
        return (f(x + Δ, y, z) - f(x, y, z)) / Δ
    catch
        try
            (f(x, y, z) - f(x - Δ, y, z)) / Δ
        catch
            error("f is not defined on [(x - Δ, y, z), (x + Δ, y, z)]")
        end
    end
end

function partial_y(f, x, y, z)
    """ Evaluate the partial derivative of f with respect to y at (x, y, z) 
    Parameters: f:R³ -> R, (x, y, z) ∈ R³
    Returns: ∂f/∂y
    """
    try
        return (f(x, y + Δ, z) - f(x, y, z)) / Δ
    catch
        try
            (f(x, y, z) - f(x, y - Δ, z)) / Δ
        catch
            error("f is not defined on [(x, y - Δ, z), (x, y + Δ, z)]")
        end
    end
end

function partial_z(f, x, y, z)
    """ Evaluate the partial derivative of f with respect to z at (x, y, z) 
    Parameters: f:R³ -> R, (x, y, z) ∈ R³
    Returns: ∂f/∂z
    """
    try
        return (f(x, y, z + Δ) - f(x, y, z)) / Δ
    catch
        try
            (f(x, y, z) - f(x, y, z - Δ)) / Δ
        catch
            error("f is not defined on [(x, y, z - Δ), (x, y, z + Δ)]")
        end
    end
end

function triple_integral(f, xi, xf, yi, yf, zi, zf)
    """ Evaluate the definite triple integral of f from xi to xf, from yi to yf, and from zi to zf
    Parameters: f:R³ -> R, a,b,c,d,e,f ∈ R
    Returns: ∭ f dxdydz
    """
    zs = []
    for z in zi:Δ:zf
        append!(zs, double_integral((x, y) -> f(x, y, z), xi, xf, yi, yf))
    end
    return sum(zs * Δ)
end

function line_integral(f, x, y, z, a, b)
    """ Evaluate the line integral of a scalar field f along the curve (x(t), y(t), z(t)) from t = a to t = b 
    Parameters: f:R³ -> R, x:R -> R, y:R -> R, z:R -> R, a,b ∈ R
    Returns: ∫ f ds
    """
    h(t) = f(x(t), y(t), z(t)) * hypot(derivative(x, t), derivative(y, t), derivative(z, t))
    return integral(h, a, b)
end

function path_integral(F, x, y, z, a, b)
    """ Evaluate the line integral of a vector field F along the curve (x(t), y(t), z(t)) from t = a to t = b 
    Parameters: F:R³ -> R³, x:R -> R, y:R -> R, z:R -> R, a,b ∈ R
    Returns: ∫ F ⋅ ds
    """
    ds(t) = [derivative(x, t), derivative(y, t), derivative(z, t)]
    h(t) = LinearAlgebra.dot(F(x(t), y(t), z(t)), ds(t))
    return integral(h, a, b)
end

function surface_integral(F, x, y, z, ti, tf, si, sf)
    """ Evaluate the flux of a vector field F through the surface x(t, s), y(t, s), z(t, s) from t = a to t = b and s = c to s = d
    Parameters: F:R³ -> R³, x:R -> R, y:R -> R, z:R -> R, a,b,c,d ∈ R
    Returns: ∬ F ⋅ dA
    """
    ∂t(t, s) = [partial_x(x, t, s), partial_x(y, t, s), partial_x(z, t, s)]
    ∂s(t, s) = [partial_y(x, t, s), partial_y(y, t, s), partial_y(y, t, s)]
    dA(t, s) = LinearAlgebra.cross(∂t(t, s), ∂s(t, s))
    f(t, s) = F(x(t, s), y(t, s), z(t, s))
    h(t, s) = LinearAlgebra.dot(f(t, s), dA(t, s))
    return double_integral(h, ti, tf, si, sf)
end

function gradient(f, x, y, z)
    """ Evaluate the gradient of f at (x, y, z) 
    Parameters: f:R³ -> R, (x, y, z) ∈ R³
    Returns: ∇f
    """
    return [partial_x(f, x, y, z), partial_y(f, x, y, z), partial_z(f, x, y, z)]
end

function divergence(F, x, y, z)
    """ Evaluate the divergence of F at (x, y, z) 
    Parameters: F:R³ -> R³, (x, y, z) ∈ R³
    Returns: ∇ ⋅ F
    """
    return partial_x(F, x, y, z)[1] + partial_y(F, x, y, z)[2] + partial_z(F, x, y, z)[3]
end

function curl(F, x, y, z)
    """ Evaluate the curl of F at (x, y, z) 
    Parameters: F:R³ -> R³, (x, y, z) ∈ R³
    Returns: ∇ × F
    """
    i = partial_z(F, x, y, z)[2] - partial_y(F, x, y, z)[3]
    j = partial_z(F, x, y, z)[1] - partial_x(F, x, y, z)[3]
    k = partial_y(F, x, y, z)[1] - partial_x(F, x, y, z)[2]
    return [i, -j, k]
end

function laplacian(f, x, y, z)
    """ Evaluate the Laplacian of f at (x, y, z) 
    Parameters: F:R³ -> R, (x, y, z) ∈ R³
    Returns: ∇²f
    """
    return divergence((u, v, w) -> gradient(f, u, v, w), x, y, z)
end

function vector_laplacian(F, x, y, z)
    """ Evaluate the vector Laplacian of F at (x, y, z) 
    Parameters: F:R³ -> R³, (x, y, z) ∈ R³
    Returns: ∇²F
    """
    g(x, y, z) = gradient((u, v, w) -> divergence(F, u, v, w), x, y, z)
    h(x, y, z) = curl((u, v, w) -> curl(F, u, v, w), x, y, z)
    return g(x, y, z) - h(x, y, z)
end