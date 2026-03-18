"""
TensorAlgebra.jl

This Julia package allows for tensor algebra and calculus using the Einstein summation convention and
symbolic indexing. It also includes tools for differential geometry, like the metric tensor, Christoffel
symbols, and covariant derivative.

# Types
Tensor{T, R} - An arbitrary rank R (m, n)-tensor of a type T
KroneckerDelta - The Kronecker Delta δ
LeviCivita - The Levi-Civita Symbol ε
PartialDerivative - The partial derivative operator ∂
CovariantDerivative - The covariant derivative operator ∇

# Functions
**General**
Tensor() - Constructor for Tensor{T, R}
getindex() - Einstein convention indexing
**Algebra**
⊗ - Tensor product
* - Tensor scaling and contraction
+ - Tensor addition
- - Tensor subtraction
⋅ - Dot product for (1, 0)-tensors
**Geometry**
metric() - Metric tensor from a basis
inv() - Invert a (2, 0) or (0, 2)-tensor
christoffel() - Compute the Levi-Civita Connection coefficients

# Examples
Defining a tensor
```
julia> L = Tensor([[1, 2]', [3, -1]'])
Tensor{Int64, 2}([1 2; 3 -1], (:contra, :co))
```
Tensor contraction
```
julia> v = Tensor([4, -2]); w = Tensor([1, 1]')
julia> v[:i] * w[:i]
2
```
Finding the metric tensor
```
julia> v = Tensor([3, 1]); u = Tensor([-1, 2])
julia> metric((u, v))
Tensor{Int64, 2}([5 -1; -1 10], (:co, :co))
```

contraflux
3/16/2026
"""

using LinearAlgebra
using Symbolics

"""
An arbitrary rank R (m, n)-tensor of a type T, where R = m + n

# Fields
data::Array{T, R}
    - The data in the tensor

variance::NTuple{R, Symbol}
    - A tuple of :contra and :co denoting the transformation rules for each index.
"""
struct Tensor{T, R}
    data::Array{T, R}
    variance::NTuple{R, Symbol}
end

"""
An indexed rank R (M, N)-tensor for type T.

Returned by getindex(A::Tensor, indices...) and getindex(A::PartialIndexedTensor, indices...)

# Fields
tensor::Tensor{T, R}
    - The tensor that has been indexed

contravariant::NTuple{M, Symbol}
    - The collection of symbols representing the contravariant indices

covariant::NTuple{N, Symbol}
    - The collection of symbols representing the covariant indices

# Examples
An indexed (1, 2) tensor
```
julia> M = Tensor([[[1, 2]', [3, 4]'], [[5, 6]', [7, 8]']]')
Tensor{Int64, 3}([1 3; 5 7;;; 2 4; 6 8], (:co, :contra, :co))
julia> M[:i][:j,:k]
IndexedTensor{Int64, 3, 1, 2}(Tensor{Int64, 3}..., (:i,), (:j, :k))
```
"""
struct IndexedTensor{T, R, M, N}
    tensor::Tensor{T, R}
    contravariant::NTuple{M, Symbol}
    covariant::NTuple{N, Symbol}
end

"""
Internal. Intermediate between Tensor{T, R} and IndexedTensor{T, R, M, N}, without specified
covariant indices.

Returned by getindex(A::Tensor, indices...).

# Fields
tensor::Tensor{T, R}
    - The tensor that has been indexed

contravariant::NTuple{M, Symbol}
    - The collection of symbols representing the contravariant indices
"""
struct PartialIndexedTensor{T, R, M}
    tensor::Tensor{T, R}
    contravariant::NTuple{M, Symbol}
end

"""
The Kronecker Delta δ

Contracts with tensors as δ[:i, :j], returning 1 if i = j and 0 otherwise.

# Examples
```
julia> L = Tensor([[1, 2]', [3, 4]'])
julia> δ = KroneckerDelta()
julia> L[:i][:j] * δ[:i, :k]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([1 3; 2 4], (:co, :contra)), (:k,), (:j,))
```
"""
struct KroneckerDelta
end

"""
Indexed Kronecker Delta

Returned by getindex(δ::KroneckerDelta, indices...)

# Fields
indices::NTuple{2, Symbol}
    - The collection of symbols representing the indices

# Examples
```
julia> δ = KroneckerDelta()
julia> δ[:i, :j]
IndexedKroneckerDelta((:i, :j))
```
"""
struct IndexedKroneckerDelta
    indices::NTuple{2, Symbol}
end

"""
The Levi-Civita Symbol ε

Contracts with tensors as ε[:i, :j, :k, ...], returning 0 if any indices are repeated, 1 if
(i, j, k...) is an even permutation, and -1 if it is an odd permutation.

# Examples
```
julia> v = Tensor([2, 1]); u = Tensor([-3, 2])
julia> ε = LeviCivita()
julia> v[:i] * u[:j] * ε[:i, :j]
7
```
"""
struct LeviCivita
end

"""
Indexed Levi-Civita Symbol

Returned by getindex(ε::LeviCivita, indices...)

# Fields
indices::NTuple{N, Symbol}
    - The collection of symbols representing the indices

# Examples
```
julia> ε = LeviCivita()
julia> ε[:i, :j, :k]
IndexedLeviCivita{3}((:i, :j, :k))
```
"""
struct IndexedLeviCivita{N}
    indices::NTuple{N, Symbol}
end

"""
The Partial Derivative Operator ∂

Contracts with tensors as ∂[:i], generating the partial derivative across the coordinates indexed by i

# Fields
coordinates::NTuple{N, Num}
    - The coordinates to differentiate with respect to

# Examples
```
julia> @variables u v
julia> x = Tensor([u^2, v])
julia> ∂ = PartialDerivative((u, v))
julia> ∂[:k] * x[:k]
1 + 2u
```
"""
struct PartialDerivative{N}
    coordinates::NTuple{N, Num}
end

"""
The Indexed Partial Derivative Operator

Returned by getindex(∂::PartialDerivative, indices...)

# Fields
partial::PartialDerivative
    - The underlying partial derivative
index::Symbol
    - The index of the coordinate

# Examples
```
julia> @variables u v
julia> ∂ = PartialDerivative((u, v))
julia> ∂[:i]
IndexedPartialDerivative(PartialDerivative{2}((u, v)), :i)
```
"""
struct IndexedPartialDerivative
    partial::PartialDerivative
    index::Symbol
end

"""
The Covariant Derivative Operator ∇

Contracts with tensors as ∇[:i], generating the covariant derivative across the coordinates indexed by i

# Fields
connection::Tensor
    - A (1, 2) tensor containing the connection coefficients
partial::PartialDerivative
    - The partial differentiation operator

# Examples
```
julia> @variables u v
julia> basis = (Tensor([1, 0]), Tensor([0, 1]))
julia> x = Tensor([u^2, v])
julia> Γ = christoffel((u, v), basis)
julia> ∂ = PartialDerivative((u, v))
julia> ∇ = CovariantDerivative(Γ.tensor, ∂)
julia> ∇[:k] * x[:k]
1 + 2u
```
"""
struct CovariantDerivative
    connection::Tensor
    partial::PartialDerivative
end

"""
The Indexed Covariant Derivative Operator

Returned by getindex(∇::CovariantDerivative, indices...)

# Fields
covariant::CovariantDerivative
    - The underlying covariant derivative
index::Symbol
    - The index of the coordinate

# Examples
```
julia> @variables u v
julia> basis = (Tensor([1, 0]), Tensor([0, 1]))
julia> Γ = christoffel((u, v), basis)
julia> ∂ = PartialDerivative((u, v))
julia> ∇ = CovariantDerivative(Γ.tensor, ∂)
julia> ∇[:k]
IndexedCovariantDerivative(CovariantDerivative(Tensor{Num, 3}..., PartialDerivative{2}...), :k)
```
"""
struct IndexedCovariantDerivative
    covariant::CovariantDerivative
    index::Symbol
end

"""
Wrapper to create rank R (m, n)-tensors with nested vectors and adjoints.

Each vector corresponds to a contravariant index, and each adjoint corresponds to a covariant index.

# Examples
A vector, (1, 0) tensor
```
julia> Tensor([1, 3])
Tensor{Int64, 1}([1, 3], (:contra,))
```
A covector, (0, 1) tensor
```
julia> Tensor([2, -5]')
Tensor{Int64, 1}([2, -5], (:co,))
```
The 90° rotation map, (1, 1) tensor
```
julia> Tensor([[0, -1]', [1, 0]'])
Tensor{Int64, 2}([0 -1; 1 0], (:contra, :co))
```
The identity metric, (0, 2) tensor
```
julia> Tensor([[1, 0]', [0, 1]']')
Tensor{Int64, 2}([1 0; 0 1], (:co, :co))
```
"""
function Tensor(data)
    variance = get_variance(data, [])
    cleaned_data = clean_data(data)
    permuted_data = permutedims(cleaned_data, Tuple([i for i in length(variance):-1:1]))
    return Tensor(permuted_data, Tuple(variance))
end

"""
Internal. Collects nested vectors and adjoints into an array.
"""
function clean_data(data)
    if isa(data, Adjoint)
        return clean_data(parent(data))
    elseif isa(data, Vector)
        return stack([clean_data(data[i]) for i in eachindex(data)])
    else
        return data
    end
end

"""
Internal. Compiles nested vectors and adjoints into variances, interpreting vectors as
contravariant and adjoints as covariant.
"""
function get_variance(data, variance)
    if isa(data, Adjoint)
        push!(variance, :co)
        return get_variance(parent(data)[1], variance)
    elseif isa(data, Vector)
        push!(variance, :contra)
        return get_variance(data[1], variance)
    else
        return variance
    end
end

"""
Index a tensor with either integer or symbolic indices.
If given integer indices, provides the contravariant indices in the first bracket and covariant indices in the second bracket, returning a scalar or array.
If given symbolic indices, returns an IndexedTensor or PartialIndexedTensor for use in Einstein summation via *.

# Arguments
A::Tensor
    - The tensor to index

indices...
    - Either integer or symbolic indices corresponding to the contravariant indices

# Examples
```
julia> L = Tensor([[1, 2]', [3, 4]'])
julia> L[1][2]
2
julia> L[:i][:j]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}..., (:i,), (:j,))
```
"""
function Base.getindex(A::Tensor, indices...)
    a = A.data
    slicing = []
    count = 0
    for i in eachindex(A.variance)
        if A.variance[i] == :co
            push!(slicing, Colon())
        else
            count += 1
            push!(slicing, indices[count])
        end
    end
    # If the indices are symbolic, return indexed tensors
    if all(isa.(indices, Symbol))
        if (count == 0) # If it's purely covariant
            return IndexedTensor(A, (), indices)
        elseif count == length(A.variance) # If it's purely contravariant
            return IndexedTensor(A, indices, ())
        end
        return PartialIndexedTensor(A, indices)
    end
    # If the indices are integers, return the sliced data
    if (count == 0) # If it's purely covariant
        return a[indices...]
    elseif (count == length(A.variance)) # If it's purely contravariant
        return a[slicing...]
    end
    return a[slicing...]
end

"""
Internal. Completes the covariant half of two-bracket symbolic indexing on a Tensor, returning a fully labeled IndexedTensor.

If any index appears as both contravariant and covariant, contract over that index.
"""
function Base.getindex(A::PartialIndexedTensor, indices...)
    duplicates = intersect(A.contravariant, indices)
    if isempty(duplicates)
        return IndexedTensor(A.tensor, A.contravariant, indices)
    end
    return self_contract(IndexedTensor(A.tensor, A.contravariant, indices), duplicates)
end

"""
Index a Kronecker Delta with either integer or symbolic indices.
If given integer indices, returns the Kronecker Delta evaluated on those indices.
If given symbolic indices, returns an IndexedKroneckerDelta for use in Einstein summation via *.

# Arguments
δ::KroneckerDelta
    - The Kronecker Delta to index

indices...
    - Either integer or symbolic indices
"""
function Base.getindex(δ::KroneckerDelta, indices...)
    if all(isa.(indices, Symbol))
        return IndexedKroneckerDelta((indices[1], indices[2]))
    end
    return indices[1] == indices[2] ? 1 : 0
end

"""
Index a Levi-Civita Symbol with either integer or symbolic indices.
If given integer indices, returns the Levi-Civita Symbol evaluated on those indices.
If given symbolic indices, returns an IndexedLeviCivita for use in Einstein summation via *.

# Arguments
ε::LeviCivita
    - The Levi-Civita Symbol to index

indices...
    - Either integer or symbolic indices
"""
function Base.getindex(ε::LeviCivita, indices...)
    if all(isa.(indices, Symbol))
        return IndexedLeviCivita((indices))
    end
    id = Matrix(I, length(indices), length(indices))
    return sign(det(id[collect(indices), :]))
end

"""
Index a Partial Derivative Operator with symbolic indices.

Returns an IndexedPartialDerivative for use in Einstein summation via *.

# Arguments
∂::PartialDerivative
    - The Partial Derivative Operator to index

indices...
    - The symbolic index
"""
function Base.getindex(∂::PartialDerivative, indices...)
    if all(isa.(indices, Symbol))
        return IndexedPartialDerivative(∂, indices[1])
    end
    return nothing
end

"""
Index a Covariant Derivative Operator with symbolic indices.

Returns an IndexedCovariantDerivative for use in Einstein summation via *.

# Arguments
∇::CovariantDerivative
    - The Covariant Derivative Operator to index

indices...
    - The symbolic index
"""
function Base.getindex(∇::CovariantDerivative, indices...)
    if all(isa.(indices, Symbol))
        return IndexedCovariantDerivative(∇, indices[1])
    end
    return nothing
end

"""
Computes the tensor product of two tensors

Given an (m, n) tensor A and a (p, q) tensor B, returns an (m + p, n + q) tensor

# Examples
A vector times a covector, returning a linear map
```
julia> v = Tensor([1, 2]); w = Tensor([3, 5]')
julia> v ⊗ w
Tensor{Int64, 2}([3 5; 6 10], (:contra, :co))
```
A covector and vector times a linear map, returning a (2, 2) tensor
```
julia> L = Tensor([[4, -1]', [2, -3]'])
julia> L ⊗ w ⊗ v
Tensor{Int64, 4}([12 -3; 6 -9;;; 20 -5; 10 -15;;;; 24 -6; 12 -18;;; 40 -10; 20 -30], (:contra, :co, :co, :contra))
```
"""
function ⊗(A::Tensor, B::Tensor)
    a = A.data
    b = B.data
    data = [a[i] * b[j] for j in eachindex(b) for i in eachindex(a)]
    return Tensor(reshape(data, size(a)..., size(b)...), (A.variance..., B.variance...))
end

"""
Computes the sum of two tensors

Given an (m, n) tensor A and a (m, n) tensor B with matching indices, returns an (m, n) tensor.

# Examples
```
julia> A = Tensor([[1, 2]', [3, -1]']); B = Tensor([[5, -2]', [1, 1]'])
julia> A[:i][:j] + B[:i][:j]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([6 0; 4 0], (:contra, :co)), (:i,), (:j,))
```
"""
function Base.:+(A::IndexedTensor, B::IndexedTensor)
    if Set(A.contravariant) != Set(B.contravariant)
        error("Contravariant indices must match")
    elseif Set(A.covariant) != Set(B.covariant)
        error("Covariant indices must match")
    end
    permutation_map = zeros(length(A.contravariant) + length(A.covariant))
    for A_index in eachindex(A.contravariant)
        B_index = findfirst(x -> x == A.contravariant[A_index], B.contravariant)
        A_idx = findindex(A_index, A.tensor.variance, :contra)
        B_idx = findindex(B_index, B.tensor.variance, :contra)
        permutation_map[A_idx] = B_idx
    end
    for A_index in eachindex(A.covariant)
        B_index = findfirst(x -> x == A.covariant[A_index], B.covariant)
        A_idx = findindex(A_index, A.tensor.variance, :co)
        B_idx = findindex(B_index, B.tensor.variance, :co)
        permutation_map[A_idx] = B_idx
    end
    C = permutedims(B.tensor.data, Tuple(Int.(permutation_map)))
    return IndexedTensor(Tensor(A.tensor.data .+ C, A.tensor.variance), A.contravariant, A.covariant)
end

"""
Computes the difference of two tensors

Given an (m, n) tensor A and a (m, n) tensor B with matching indices, returns an (m, n) tensor

# Examples
```
julia> A = Tensor([[1, 2]', [3, -1]']); B = Tensor([[5, -2]', [1, 1]'])
julia> A[:i][:j] - B[:i][:j]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([-4 4; 2 -2], (:contra, :co)), (:i,), (:j,))
```
"""
function Base.:-(A::IndexedTensor, B::IndexedTensor)
    return A + (-1 * B)
end

"""
Computes the contraction of two tensors A and B along the specified indices.

If no indices match, return the tensor product of A and B

# Examples
A covector acting on a vector, returning a scalar
```
julia> v = Tensor([1, 2]); w = Tensor([3, 5]')
julia> v[:i] * w[:i]
13.0
```
A linear map acting on a vector, returning a vector
```
julia> L = Tensor([[4, -1]', [2, -3]'])
julia> L[:i][:j] * v[:j]
Tensor{Float64, 1}([2.0, -4.0], (:contra,))
```
"""
function Base.:*(A::IndexedTensor, B::IndexedTensor)
    # Simple error checking to avoid repeated indices
    if !isempty(intersect(A.contravariant, B.contravariant))
        error("Repeated contravariant indices")
    elseif !isempty(intersect(A.covariant, B.covariant))
        error("Repeated covariant indices")
    end
    # Find pairs of matching contravariant and covariant indices across A and B
    pairs = find_pairs(A, B)
    # Find the free indicies (those without pairs)
    all_indices = union(A.contravariant, A.covariant, B.contravariant, B.covariant)
    contra_contractions = intersect(A.contravariant, B.covariant)
    co_contractions = intersect(A.covariant, B.contravariant)
    free_indices = setdiff(all_indices, union(contra_contractions, co_contractions))

    ranges = []
    for (a_idx, b_idx) in pairs
        if size(A.tensor.data, a_idx) != size(B.tensor.data, b_idx)
            error("Dimensions not the same size")
        end
        push!(ranges, 1:size(A.tensor.data, a_idx))
    end
    free_indices_info = []
    for free_index in free_indices
        if (free_index in A.contravariant)
            A_index = find_index(free_index, A, :contra)
            info = (tensor=:A, index=A_index, dimension=size(A.tensor.data, A_index), variance=:contra)
            push!(free_indices_info, info)
        elseif (free_index in A.covariant)
            A_index = find_index(free_index, A, :co)
            info = (tensor=:A, index=A_index, dimension=size(A.tensor.data, A_index), variance=:co)
            push!(free_indices_info, info)
        elseif (free_index in B.contravariant)
            B_index = find_index(free_index, B, :contra)
            info = (tensor=:B, index=B_index, dimension=size(B.tensor.data, B_index), variance=:contra)
            push!(free_indices_info, info)
        else
            B_index = find_index(free_index, B, :co)
            info = (tensor=:B, index=B_index, dimension=size(B.tensor.data, B_index), variance=:co)
            push!(free_indices_info, info)
        end
    end

    free_ranges = [1:info.dimension for info in free_indices_info]
    # Promote element types across both tensors so e.g. Int64 * Float64 or Int64 * Num
    # don't cause a type mismatch when accumulating into result
    T = promote_type(eltype(A.tensor.data), eltype(B.tensor.data))
    result = zeros(T, [info.dimension for info in free_indices_info]...)
    for free_index in Iterators.product(free_ranges...)
        for index in Iterators.product(ranges...)
            A_idx = Any[0 for _ in 1:ndims(A.tensor.data)]
            B_idx = Any[0 for _ in 1:ndims(B.tensor.data)]
            for (value, info) in zip(free_index, free_indices_info)
                if info.tensor == :A
                    A_idx[info.index] = value
                else
                    B_idx[info.index] = value
                end
            end
            for (value, location) in zip(index, pairs)
                A_idx[location[1]] = value
                B_idx[location[2]] = value
            end
            result[free_index...] += A.tensor.data[A_idx...] .* B.tensor.data[B_idx...]
        end
    end

    leftover_contra_indices = []
    leftover_co_indices = []
    for (index, info) in zip(free_indices, free_indices_info)
        if info.variance == :contra
            push!(leftover_contra_indices, index)
        else
            push!(leftover_co_indices, index)
        end
    end
    # If all indices are free, return the scalar product
    if isempty(pairs)
        return IndexedTensor(A.tensor ⊗ B.tensor, Tuple(leftover_contra_indices), Tuple(leftover_co_indices))
    end
    # If it's a scalar, return as a scalar
    if isempty(free_indices)
        return result[]
    end
    return IndexedTensor(Tensor(result, Tuple([info.variance for info in free_indices_info])), Tuple(leftover_contra_indices), Tuple(leftover_co_indices))
end

"""
Scales every element of a tensor A by number s.

# Examples
```
julia> A = Tensor([[1, 2]', [3, -1]'])
julia> 2 * A[:i][:j]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([2 4; 6 -2], (:contra, :co)), (:i,), (:j,))
```
"""
function Base.:*(A::IndexedTensor, s::Number)
    scaled = A.tensor.data .* s
    return IndexedTensor(Tensor(scaled, A.tensor.variance), A.contravariant, A.covariant)
end


function Base.:*(s::Number, A::IndexedTensor)
    return A * s
end

"""
Computes the contraction of a tensor and the Kronecker Delta

# Examples
```
julia> A = Tensor([[1, 2]', [3, -1]'])
julia> δ = KroneckerDelta()
julia> A[:i][:j] * δ[:j, :k]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([1 2; 3 -1], (:contra, :co)), (:i,), (:k,))
julia> A[:i][:j] * δ[:j, :i]
0
```
"""
function Base.:*(A::IndexedTensor, δ::IndexedKroneckerDelta)
    pairs = find_pairs(A, δ)
    if isempty(pairs)
        error("Delta has no common indices")
    end
    (index, symbol, variance) = pairs[1]
    dim = size(A.tensor.data, index)
    δTensor = Tensor(Matrix(I, dim, dim), (:contra, :co))
    if symbol == δ.indices[1]
        δIndexedTensor = variance == :co ? δTensor[symbol][δ.indices[2]] : δTensor[δ.indices[2]][symbol]
    else
        δIndexedTensor = variance == :co ? δTensor[symbol][δ.indices[1]] : δTensor[δ.indices[1]][symbol]
    end
    return A * δIndexedTensor
end

function Base.:*(δ::IndexedKroneckerDelta, A::IndexedTensor)
    return A * δ
end

"""
Computes the contraction of a tensor and the Levi-Civita Symbol

# Examples
```
julia> v = Tensor([1, 2]); u = Tensor([-2, 1])
julia> ε = LeviCivita()
julia> v[:i] * u[:j] * ε[:i, :j]
5
```
"""
function Base.:*(A::IndexedTensor, ε::IndexedLeviCivita)
    pairs = find_pairs(A, ε)
    if isempty(pairs)
        error("Levi Civita has no common indices")
    end
    free_indices = setdiff(ε.indices, map(x -> x[3], pairs))
    dims = map(x -> size(A.tensor.data, x[1]), pairs)
    if !all(dim -> dim == dims[1], dims)
        error("Levi Civita has non-constant dimension")
    end
    append!(dims, [dims[1] for _ in free_indices])
    B = zeros(Int, dims...)
    n = max(dims[1], length(dims))
    id = Matrix(I, n, n)
    for indices in Iterators.product(ntuple(_ -> 1:dims[1], length(dims))...)
        B[collect(indices)...] = round(Int, sign(det(id[collect(indices), 1:length(dims)])))
    end
    εTensor = Tensor(B, Tuple([pairs[1][4] == :contra ? :co : :contra for _ in dims]))
    if pairs[1][4] == :contra
        εIndexedTensor = IndexedTensor(εTensor, (), ε.indices)
    else
        εIndexedTensor = IndexedTensor(εTensor, ε.indices, ())
    end
    return A * εIndexedTensor
end

function Base.:*(ε::IndexedLeviCivita, A::IndexedTensor)
    return A * ε
end

"""
Computes the element-wise partial derivative of a tensor A with respect to the coordinates of ∂.

Given an (m, n) tensor A, returns an (m, n+1) tensor. If indices are repeated, contract over them.

# Examples
```
julia> @variables u v
julia> A = Tensor([[u, 2v]', [3u, v^2]'])
julia> ∂ = PartialDerivative((u, v))
julia> ∂[:k] * A[:i][:j]
IndexedTensor{Num, 3, 1, 2}(Tensor{Num, 3}(Num[1 0; 3 0;;; 0 2; 0 2v], (:contra, :co, :co)), (:i,), (:j, :k))
julia> ∂[:k] * A[:k][:j]
IndexedTensor{Num, 1, 0, 1}(Tensor{Num, 1}(Num[1, 2v], (:co,)), (), (:j,))
```
"""
function Base.:*(∂::IndexedPartialDerivative, A::IndexedTensor)
    Bs = []
    for coordinate in ∂.partial.coordinates
        push!(Bs, map(a -> expand_derivatives(Differential(coordinate)(a)), A.tensor.data))
    end
    B = stack(Bs)
    C = IndexedTensor(Tensor(B, (A.tensor.variance..., :co)), A.contravariant, (A.covariant..., ∂.index))
    duplicates = union(intersect(A.contravariant, [∂.index]), intersect(A.covariant, [∂.index]))
    if isempty(duplicates)
        return C
    end
    return self_contract(C, duplicates)
end

"""
Computes the covariant derivative of a tensor A with respect to the coordinates of ∇.

Given an (m, n) tensor A, returns an (m, n+1) tensor. If indices are repeated, contract over them.
Accounts for the change in the basis vectors from the connection coefficients.

# Examples
```
julia> @variables u v
julia> basis = (Tensor([u, 0]), Tensor([0, v]))
julia> ∂ = PartialDerivative((u, v))
julia> Γ = christoffel((u, v), basis)
julia> x = Tensor([2u + v^2, 1v])
julia> ∇ = CovariantDerivative(Γ.tensor, ∂)
julia> ∇[:k] * x[:i]
IndexedTensor{Num, 2, 1, 1}(Tensor{Num, 2}(Num[2 + (2u + v^2) / u 2v; 0.0 2.0], (:contra, :co)), (:i,), (:k,))
```
"""
function Base.:*(∇::IndexedCovariantDerivative, A::IndexedTensor)
    Γ = ∇.covariant.connection
    ∂ = ∇.covariant.partial
    B = ∂[∇.index] * A
    for index in A.contravariant
        dummy_index = Symbol("dummy_$index")
        C_contravariant = collect(A.contravariant)
        contra_index = findfirst(x -> x == index, A.contravariant)
        C_contravariant[contra_index] = dummy_index
        ΓIndexed = Γ[index][dummy_index, ∇.index]
        C = IndexedTensor(A.tensor, Tuple(C_contravariant), A.covariant)
        B += ΓIndexed * C
    end
    for index in A.covariant
        dummy_index = Symbol("dummy_$index")
        C_covariant = collect(A.covariant)
        co_index = findfirst(x -> x == index, A.covariant)
        C_covariant[co_index] = dummy_index
        ΓIndexed = Γ[dummy_index][index, ∇.index]
        C = IndexedTensor(A.tensor, A.contravariant, Tuple(C_covariant))
        B -= ΓIndexed * C
    end
    return B
end

"""
Internal. Contracts a tensor A along duplicate indices by contracting with the Kronecker Delta δ
"""
function self_contract(A::IndexedTensor, duplicates)
    pairs = []
    B_covariant = collect(A.covariant)
    for i in eachindex(duplicates)
        index = duplicates[i]
        dummy_index = Symbol("dummy_$i")
        co_index = findfirst(x -> x == index, A.covariant)
        B_covariant[co_index] = dummy_index
        push!(pairs, (index, dummy_index))
    end
    B = IndexedTensor(A.tensor, A.contravariant, Tuple(B_covariant))
    δ = KroneckerDelta()
    for (index, dummy_index) in pairs
        B = B * δ[index, dummy_index]
    end
    return B
end

"""
Internal. Converts a symbolic index to its index in a tensor's data array.
"""
function find_index(target, A::IndexedTensor, variance)
    space = variance == :contra ? A.contravariant : A.covariant
    contra_index = findfirst(x -> x == target, space)
    index = findindex(contra_index, A.tensor.variance, variance)
    return index
end


"""
Internal. Finds symbolic pairs in the contravariant and covariant indices of two tensors A and B.
Returns matching indices in the data arrays of A and B.
"""
function find_pairs(A::IndexedTensor, B::IndexedTensor)
    pairs = []
    for index in intersect(A.contravariant, B.covariant)
        A_index = find_index(index, A, :contra)
        B_index = find_index(index, B, :co)
        push!(pairs, (A_index, B_index))
    end
    for index in intersect(A.covariant, B.contravariant)
        A_index = find_index(index, A, :co)
        B_index = find_index(index, B, :contra)
        push!(pairs, (A_index, B_index))
    end
    return pairs
end

"""
Internal. Finds symbolic pairs in the contravariant and covariant indices a tensor A and the
Kronecker Delta δ. Returns matching indices in the data arrays of A and δ, including the variance
of the index.
"""
function find_pairs(A::IndexedTensor, δ::IndexedKroneckerDelta)
    pairs = []
    for index in intersect(A.contravariant, δ.indices)
        A_index = find_index(index, A, :contra)
        push!(pairs, (A_index, index, :contra))
    end
    for index in intersect(A.covariant, δ.indices)
        A_index = find_index(index, A, :co)
        push!(pairs, (A_index, index, :co))
    end
    return pairs
end

"""
Internal. Finds symbolic pairs in the contravariant and covariant indices a tensor A and the
Levi-Civita Symbol ε. Returns matching indices in the data arrays of A and ε, including the 
symbolic index and the variance of the index.
"""
function find_pairs(A::IndexedTensor, ε::IndexedLeviCivita)
    pairs = []
    for index in intersect(A.contravariant, ε.indices)
        A_index = find_index(index, A, :contra)
        ε_index = findfirst(x -> x == index, ε.indices)
        push!(pairs, (A_index, ε_index, index, :contra))
    end
    for index in intersect(A.covariant, ε.indices)
        A_index = find_index(index, A, :co)
        ε_index = findfirst(x -> x == index, ε.indices)
        push!(pairs, (A_index, ε_index, index, :co))
    end
    return pairs
end

"""
Internal. Finds the index of the nth occurrance of key in a list.
"""
function findindex(desired_index, variance, key)
    count = 0
    var_index = 0
    for i in eachindex(variance)
        var_index = i
        if variance[i] == key
            count += 1
        end
        if count == desired_index
            break
        end
    end
    return var_index
end

"""
Computes the inverse of a (2, 0) or (0, 2) tensor (the metric or inverse metric)
"""
function LinearAlgebra.inv(A::Tensor)
    if length(A.variance) != 2
        error("Must be a rank 2 tensor")
    elseif A.variance[1] != A.variance[2]
        error("Must either a (2, 0) or (0, 2) tensor")
    end
    mat = eltype(A.data) <: Num ? Matrix{Num}(A.data) : Matrix{Float64}(A.data)
    if A.variance[1] == :co
        return Tensor(inv(mat), (:contra, :contra))
    else
        return Tensor(inv(mat), (:co, :co))
    end
end

"""
Define the inner product on two (1, 0) tensors.

Currently the standard inner product.

# Examples
```
julia> v = Tensor([1, 2]); w = Tensor([3, -1])
Tensor{Int64, 1}([3, -1], (:contra,))
julia> v ⋅ w
1
```
"""
function LinearAlgebra.:⋅(A::Tensor, B::Tensor)
    if A.variance != (:contra,) || B.variance != (:contra,)
        error("A and B must both be (1, 0) tensors")
    end
    return sum([A.data[i] * B.data[i] for i in eachindex(A.data)])
end

"""
Compute the metric tensor g from a vector basis

# Examples
```
julia> basis = (Tensor([1, 2]), Tensor([3, -1]))
julia> g = metric(basis)
Tensor{Int64, 2}([5 1; 1 10], (:co, :co))
```
"""
function metric(basis)
    g = [basis[i] ⋅ basis[j] for i in eachindex(basis), j in eachindex(basis)]
    return Tensor(g, (:co, :co,))
end

"""
Compute the Christoffel Symbols Γ for the Levi-Civita Connection from the coordinates and a basis

Returns a (1, 2) IndexedTensor with indices [:l][:j, :k]

# Examples
```
julia> @variables u v
julia> basis = (Tensor([u, 0]), Tensor([0, v]))
julia> christoffel((u, v), basis)
IndexedTensor{Num, 3, 1, 2}(Tensor{Num, 3}..., (:l,), (:j, :k))
```
"""
function christoffel(coordinates, basis)
    ∂ = PartialDerivative(coordinates)
    g = metric(basis)
    G = inv(g)
    T1 = ∂[:k] * g[:r, :j]
    T2 = ∂[:j] * g[:r, :k]
    T3 = ∂[:r] * g[:j, :k]
    return 0.5 * G[:l, :r] * (T1 + T2 - T3)
end