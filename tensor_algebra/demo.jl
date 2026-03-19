include("tensors.jl")

"""
Tensor Algebra
"""

# Defining tensors
v = Tensor([1, 2]) # A vector (0, 1)-tensor
ω = Tensor([3, -1]') # A covector (one-form) (1, 0)-tensor
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
L[:i][:j] * v[:j] # Multiplying a linear map and a vector -> (0, 1)-tensor
ω[:i] * L[:i][:j] * v[:j] # Multipling a covector, linear map, and vector -> (0, 0)-tensor

# Combining tensors with the tensor product
M = v ⊗ ω ⊗ v # Two vectors combined with a covector -> (2, 1)-tensor

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
v[:i] * u[:j] * ε[:i, :j] # Wedge product of two vectors

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
v = Tensor([u^2, v]) # A non-constant vector (1, 0)-tensor
∇[:k] * v[:i] # The covariant derivative of v, (1, 1)-tensor

# Lie brackets
X = Tensor([u^2 + 1, -2v]) # A (1, 0)-tensor
Y = Tensor([v, 3 - v]) # A (1, 0)-tensor
lie(X, Y, ∂) # The Lie bracket [X, Y], (1, 0)-tensor

# Riemann and Ricci Curvature Tensors, and the Ricci Scalar
basis = (Tensor([1, 0]), Tensor([0, sin(v)]))
riemann((u, v), basis) # The Riemann Curvature Tensor, (1, 3)-tensor
ricci((u, v), basis) # The Ricci Curvature Tensor, (0, 2)-tensor
ricci_scalar((u, v), basis) # The Ricci Scalar, (0, 0)-tensor