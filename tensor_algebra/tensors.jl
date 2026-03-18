"""
TensorAlgebra.jl

This Julia package allows for general operations with tensors,
including the tensor product and contraction.

# Types
Tensor{T, N} - General (m, n) tensor of objects with type T
KroneckerDelta - The Kronecker Delta δ
LeviCivita - The Levi-Civita Symbol ε

# Functions
Tensor - Wrapper for Tensor{T, N}, takes nested vectors (contravariant indices) and adjoints (covariant indices)
getindex - Allows indexing independently through contra and covariant indices, and taking the trace
⊗ - Tensor product
* - Tensor contraction

# Examples
```
julia> v = Tensor([1, 2])
Tensor{Int64, 1}([1, 2], (:contra,))
julia> w = Tensor([3, -1]')
Tensor{Int64, 1}([3, -1], (:co,))
julia> v[:i] * w[:i]
1.0
julia> L = v ⊗ w
Tensor{Int64, 2}([3 -1; 6 -2], (:contra, :co))
julia> w[:i] * L[:i][:j] * 5 * v[:j]
5.0
```

contraflux
3/16/2026
"""

using LinearAlgebra

"""
General (m, n) tensor of objects with type T

# Fields
data::Array{T, N}
    - The data in the tensor

variance::NTuple{N, Symbol}
    - A tuple of :contra and :co denoting the transformation rules for each index.
"""
struct Tensor{T, N}
    data::Array{T, N}
    variance::NTuple{N, Symbol}
end

"""
Indexed (M, N) tensor of objects with type T

# Fields
tensor::Tensor{T, R}
    - The tensor that has been indexed

contravariant::NTuple{M, Symbol}
    - The collection of symbols representing the contravariant indices

covariant::NTuple{N, Symbol}
    - The collection of symbols representing the covariant indices

# Examples
An indexed vector, (1, 0) tensor
```
julia> v = Tensor([1, 3])
Tensor{Int64, 1}([1, 3], (:contra,))
julia> v[:i]
IndexedTensor{Int64, 1, 1, 0}(Tensor{Int64, 1}([1, 3], (:contra,)), (:i,), ())
```
An indexed covector, (0, 1) tensor
```
julia> w = Tensor([2, 5]')
Tensor{Int64, 1}([2, 5], (:co,))
julia> w[:j]
IndexedTensor{Int64, 1, 0, 1}(Tensor{Int64, 1}([2, 5], (:co,)), (), (:j,))
```
An indexed (1, 2) tensor
```
julia> M = Tensor([[[1, 2]', [3, 4]'], [[5, 6]', [7, 8]']]')
Tensor{Int64, 3}([1 3; 5 7;;; 2 4; 6 8], (:co, :contra, :co))
julia> M[:i][:j,:k]
IndexedTensor{Int64, 3, 1, 2}(Tensor{Int64, 3}([1 3; 5 7;;; 2 4; 6 8], (:co, :contra, :co)), (:i,), (:j, :k))
```
"""
struct IndexedTensor{T, R, M, N}
    tensor::Tensor{T, R}
    contravariant::NTuple{M, Symbol}
    covariant::NTuple{N, Symbol}
end

struct SplitTensor{T, N}
    data::Array{T, N}
end

struct PartialIndexedTensor{T, R, M}
    tensor::Tensor{T, R}
    contravariant::NTuple{M, Symbol}
end

"""
The Kronecker Delta δ
"""
struct KroneckerDelta
end

"""
Indexed Kronecker Delta

# Fields
indices::NTuple{2, Symbol}
    - The collection of symbols representing the indices

# Examples
An Kronecker Delta
```
julia> δ = KroneckerDelta()
KroneckerDelta()
julia> δ[:i, :j]
IndexedKroneckerDelta((:i, :j))
```
"""
struct IndexedKroneckerDelta
    indices::NTuple{2, Symbol}
end


"""
The Levi-Civita Symbol ε
"""
struct LeviCivita
end

"""
Indexed Levi-Civita Symbol

# Fields
indices::NTuple{N, Symbol}
    - The collection of symbols representing the indices

# Examples
```
julia> ε = LeviCivita()
LeviCivita()
julia> ε[:i, :j, :k]
IndexedLeviCivita{3}((:i, :j, :k))
```
"""
struct IndexedLeviCivita{N}
    indices::NTuple{N, Symbol}
end

"""
Simple constructor for (m, n) tensors of objects with type T

Takes in nested vectors and adjoints, inferring co and contravariance.

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

function clean_data(data)
    if isa(data, Adjoint)
        return clean_data(parent(data))
    elseif isa(data, Vector)
        return stack([clean_data(data[i]) for i in eachindex(data)])
    else
        return data
    end
end

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
If given integer indices, provides the contravariant indices in the first bracket and covariant indices in the second bracket, returning a scalar.
If given symbolic indices, returns an IndexedTensor or PartialIndexedTensor for use in Einstein summation via *.

# Arguments
A::Tensor
    - The tensor to index

indices...
    - Either integer or symbolic indices corresponding to the contravariant indices

# Examples
```
julia> L = Tensor([[1, 2]', [3, 4]'])
Tensor{Int64, 2}([1 3; 2 4], (:contra, :co))
julia> L[1][2]
3
julia> L[:i][:j]
IndexedTensor{Int64, 2, 1, 1}(Tensor{Int64, 2}([1 3; 2 4], (:contra, :co)), (:i,), (:j,))
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
    return SplitTensor(a[slicing...])
end

"""
Internal. Completes the covariant half of two-bracket integer indexing on a Tensor.
"""
function Base.getindex(A::SplitTensor, indices...)
    a = A.data
    return a[indices...]
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
    pairs = []
    B_covariant = collect(indices)
    for i in eachindex(duplicates)
        index = duplicates[i]
        dummy_index = Symbol("dummy_$i")
        co_index = findfirst(x -> x == index, indices)
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
Index a Kronecker Delta with either integer or symbolic indices.
If given integer indices, returns the Kronecker Delta evaluated on those indices.
If given symbolic indices, returns an IndexedKroneckerDelta for use in Einstein summation via *.

# Arguments
δ::KroneckerDelta
    - The Kronecker Delta to index

indices...
    - Either integer or symbolic indices

# Examples
```
julia> δ = KroneckerDelta()
KroneckerDelta()
julia> δ[1, 2]
0
julia> δ[3, 3]
1
julia> δ[:i, :j]
IndexedKroneckerDelta((:i, :j))
```
"""
function Base.getindex(δ::KroneckerDelta, indices...)
    if all(isa.(indices, Symbol))
        return IndexedKroneckerDelta((indices[1], indices[2]))
    else
        if indices[1] == indices[2]
            return 1
        else
            return 0
        end
    end
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

# Examples
```
julia> ε = LeviCivita()
LeviCivita()
julia> ε[1, 2, 3]
1.0
julia> ε[1, 3, 2]
-1.0
julia> ε[:i, :j, :k]
IndexedLeviCivita{3}((:i, :j, :k))
```
"""
function Base.getindex(ε::LeviCivita, indices...)
    if all(isa.(indices, Symbol))
        return IndexedLeviCivita((indices))
    else
        id = Matrix(I, length(indices), length(indices))
        return sign(det(id[collect(indices), :]))
    end
end

"""
Computes the tensor product of two tensors

Given an (m, n) tensor A and a (p, q) tensor B, returns an (m + p, n + q) tensor

# Examples
A vector times a covector, returning a linear map
```
julia> v = Tensor([1, 2]); w = Tensor([3, 5]')
Tensor{Int64, 1}([3, 5], (:co,))
julia> v ⊗ w
Tensor{Int64, 2}([3 5; 6 10], (:contra, :co))
```
A covector and vector times a linear map, returning a (2, 2) tensor
```
julia> L = Tensor([[4, -1]', [2, -3]'])
Tensor{Int64, 2}([4 -1; 2 -3], (:contra, :co))
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
Computes the contraction of two tensors along the specified indices

# Examples
A covector acting on a vector, returning a scalar
```
julia> v = Tensor([1, 2]); w = Tensor([3, 5]')
Tensor{Int64, 1}([3, 5], (:co,))
julia> v[:i] * w[:i]
13.0
```
A linear map acting on a vector, returning a vector
```
julia> L = Tensor([[4, -1]', [2, -3]'])
Tensor{Int64, 2}([4 -1; 2 -3], (:contra, :co))
julia> L[:i][:j] * v[:j]
Tensor{Float64, 1}([2.0, -4.0], (:contra,))
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
    result = zeros(eltype(A.tensor.data), [info.dimension for info in free_indices_info]...)
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

function Base.:*(A::IndexedTensor, s::Number)
    scaled = A.tensor.data .* s
    return IndexedTensor(Tensor(scaled, A.tensor.variance), A.contravariant, A.covariant)
end

function Base.:*(s::Number, A::IndexedTensor)
    return A * s
end

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

function find_index(target, A::IndexedTensor, variance)
    space = variance == :contra ? A.contravariant : A.covariant
    contra_index = findfirst(x -> x == target, space)
    index = findindex(contra_index, A.tensor.variance, variance)
    return index
end

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
    if A.variance[1] == :co
        return Tensor(inv(A.data), (:contra, :contra))
    else
        return Tensor(inv(A.data), (:co, :co))
    end
end