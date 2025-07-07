include("calc_1d.jl")

function partial_x(f, x, y)
    """ Evaluate the partial derivative of f with respect to x at (x, y) 
    Parameters: f:R² -> R, (x, y) ∈ R²
    Returns: ∂f/∂x
    """
    try
        return (f(x + Δ, y) - f(x, y)) / Δ
    catch
        try
            (f(x, y) - f(x - Δ, y)) / Δ
        catch
            error("f is not defined on [(x - Δ, y), (x + Δ, y)]")
        end
    end
end

function partial_y(f, x, y)
    """ Evaluate the partial derivative of f with respect to y at (x, y) 
    Parameters: f:R² -> R, (x, y) ∈ R²
    Returns: ∂f/∂y
    """
    try
        return (f(x, y + Δ) - f(x, y)) / Δ
    catch
        try
            (f(x, y) - f(x, y - Δ)) / Δ
        catch
            error("f is not defined on [(x, y - Δ), (x, y + Δ)]")
        end
    end
end

function double_integral(f, xi, xf, yi, yf)
    """ Evaluate the definite double integral of f from xi to xf and from yi to yf
    Parameters: f:R² -> R, a,b,c,d ∈ R
    Returns: ∬ f dxdy
    """
    ys = []
    for y in yi:Δ:yf
        append!(ys, integral(x -> f(x, y), xi, xf))
    end
    return sum(ys * Δ)
end

function line_integral(f, x, y, a, b)
    """ Evaluate the line integral of a scalar field f along the curve (x(t), y(t)) from t = a to t = b 
    Parameters: f:R² -> R, x:R -> R, y:R -> R, a,b ∈ R
    Returns: ∫ f ds
    """
    h(t) = f(x(t), y(t)) * hypot(derivative(x, t), derivative(y, t))
    return integral(h, a, b)
end

function path_integral(F, x, y, a, b)
    """ Evaluate the line integral of a vector field F along the curve (x(t), y(t)) from t = a to t = b 
    Parameters: F:R² -> R², x:R -> R, y:R -> R, a,b ∈ R
    Returns: ∫ F ⋅ ds
    """
    ds(t) = [derivative(x, t), derivative(y, t)]
    h(t) = LinearAlgebra.dot(F(x(t), y(t)), ds(t))
    return integral(h, a, b)
end

function flux_integral(F, x, y, a, b)
    """ Evaluate the flux of a vector field F through the curve (x(t), y(t)) from t = a to t = b
    Parameters: F:R² -> R², x:R -> R, y:R -> R, a,b ∈ R
    Returns: ∫ F ⋅ n̂ds
    """
    n̂ds(t) = [-derivative(y, t), derivative(x, t)]
    h(t) = LinearAlgebra.dot(F(x(t), y(t)), n̂ds(t))
    return integral(h, a, b)
end

function gradient(f, x, y)
    """ Evaluate the gradient of f at (x, y) 
    Parameters: f:R² -> R, (x, y) ∈ R²
    Returns: ∇f
    """
    return [partial_x(f, x, y), partial_y(f, x, y)]
end

function divergence(F, x, y)
    """ Evaluate the divergence of F at (x, y) 
    Parameters: F:R² -> R², (x, y) ∈ R²
    Returns: ∇ ⋅ F
    """
    return partial_x(F, x, y)[1] + partial_y(F, x, y)[2]
end

function curl(F, x, y)
    """ Evalute the curl of F at (x, y)
    Parameters: F:R² -> R², (x, y) ∈ R²
    Returns: ∇ × F
    """
    return partial_y(F, x, y)[1] - partial_x(F, x, y)[2]
end

function laplacian(f, x, y)
    """ Evaluate the Laplacian of f at (x, y, z) 
    Parameters: F:R² -> R, (x, y) ∈ R²
    Returns: ∇²f
    """
    return divergence((u, v) -> gradient(f, u, v), x, y)
end