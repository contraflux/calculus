"""
demo.jl

This Julia script demonstrates the features of tensors.jl

contraflux
3/20/2026
"""

include("tensors.jl")

"""
Tensor Algebra
"""

# Defining tensors
v = Tensor([1, 2]) # A vector (1, 0)-tensor
ω = Tensor([3, -1]') # A covector (one-form) (0, 1)-tensor
L = Tensor([[3, 2]', [1, -2]']) # A linear map (1, 1)-tensor

# Indexing tensors
v[1] # First index of v
ω[2] # Second index of ω
L[2][1] # Second contravariant index, first covariant index of L

# Scalar product
u = Tensor([2, -3])
v ⋅ u # The dot product of two vectors

# Multiplying tensors
v[:i] * ω[:i] # Multiplying a vector and covector -> (0, 0)-tensor
L[:i][:j] * v[:j] # Multiplying a linear map and a vector -> (1, 0)-tensor
ω[:i] * L[:i][:j] * v[:j] # Multipling a covector, linear map, and vector -> (0, 0)-tensor

# Combining tensors with the tensor product
M = v ⊗ ω ⊗ v # Two vectors combined with a covector -> (2, 1)-tensor

# Combining differential forms with the wedge product
α = Tensor([1, 2]') # A covector (one-form), (0, 1)-tensor
β = Tensor([-3, 1]') # Another covector (one-form), (0, 1)-tensor
γ = α ∧ β # The wedge product of two one-forms, returning a two-form (0, 2)-tensor

# Symmetrizing and antisymmetrizing
N = Tensor([1, 2]) ⊗ Tensor([-3, 1]) ⊗ Tensor([2, -4]) ⊗ Tensor([-5, -4]') ⊗ Tensor([2, -1]') # (3, 2)-tensor
symmetrize(N[:i, :j, :k][:l, :m], :i, :k) # Symmetrize across :i and :k
symmetrize(N[:i, :j, :k][:l, :m], :i, :j, :k) # Symmetrize across :i, :j, and :k
antisymmetrize(N[:i, :j, :k][:l, :m], :l, :m) # Antisymmetrize across :l, :m

# Basis operations
basis = (Tensor([1, 1]), Tensor([0, 2])) # A vector basis
g = metric(basis) # The metric tensor (0, 2)-tensor
G = inv(g) # The inverse metric tensor (2, 0)-tensor
g = metric(basis, minkowski) # The metric (0, 2)-tensor in Minkowski space with signature (-, +, +, +)

# Kronecker Delta
δ = KroneckerDelta()
g[:i, :j] * δ[:j, :k] # Index swapping i -> k
L[:m][:n] * δ[:m, :p] # Index swapping m -> p

# Levi-Civita Symbol
ε = LeviCivita()
v[:i] * u[:j] * ε[:i, :j] # Signed area between two vectors

# Hodge Star
basis = (Tensor([1, 0]), Tensor([0, 1]))
α = Tensor([1, 2]')
g = metric(basis)
⋆ = HodgeStar(g)
⋆(α)

"""
Tensor Calculus
"""

@variables u v
basis = (Tensor([2, -v]), Tensor([u^2, 1])) # A non-constant vector basis

# Partial derivatives
∂ = PartialDerivative((u, v)) # The partial derivative
T = Tensor([[u, -u]', [3v, u + v]']) # A non-constant linear map (1, 1)-tensor
∂[:k] * T[:i][:j] # The partial derivative of T, (1, 2)-tensor

# Covariant derivatives
Γ = christoffel((u, v), basis) # The Christoffel Symbols for the Levi-Civita Connection
∇ = CovariantDerivative(Γ, ∂) # The covariant derivative
x = Tensor([u^2, v]) # A non-constant vector (1, 0)-tensor
∇[:k] * x[:i] # The covariant derivative of x, (1, 1)-tensor

# Exterior derivatives
@variables x y z
∂ = PartialDerivative((x, y, z)) # The partial derivative
d = ExteriorDerivative(∂) # The exterior derivative
α = Tensor([x^2, y*z, x]') # A covector (one-form), (0, 1)-tensor
d[:k] * α[:i] # The exterior derivative of a one-form, a two-form (0, 2)-tensor

# Lie brackets
@variables u v
∂ = PartialDerivative((u, v)) # The partial derivative
X = Tensor([u^2 + 1, -2v]) # A (1, 0)-tensor
Y = Tensor([v, 3 - v]) # A (1, 0)-tensor
lie(X, Y, ∂) # The Lie bracket [X, Y], (1, 0)-tensor

# Riemann and Ricci Curvature Tensors, and the Ricci Scalar
basis = (Tensor([1, 0]), Tensor([0, sin(v)]))
riemann((u, v), basis) # The Riemann Curvature Tensor, (1, 3)-tensor
ricci((u, v), basis) # The Ricci Curvature Tensor, (0, 2)-tensor
ricci_scalar((u, v), basis) # The Ricci Scalar, (0, 0)-tensor
einstein((u, v), basis) # The Einstein Tensor, (0, 2)-tensor