using Symbolics

include("tensors.jl")

@variables a b c d
@variables x y

T = Tensor([[a, b]', [c, d]'])
v = Tensor([x, y])
T ⊗ v
T[:i][:j] * v[:j]

g = Tensor([[1., 2]', [-2, 1]']')
inv_g = inv(g)

display(g)
display(inv_g)

# println(g.data)
# println(inv_g.data)
g[:i, :j] * inv_g[:j, :k]
# inv_g[:j, :k]