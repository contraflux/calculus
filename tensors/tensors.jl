mutable struct tensor
    rank::Int
    type::Vector
    array::Array
end

function δ(i::Int, j::Int)
    i == j ? 1 : 0
end

e₁ = [1; 0]; e₂ = [0; 1]
e¹ = [1 0]; e² = [0 1]

e¹ * e₁

e₁ * e¹