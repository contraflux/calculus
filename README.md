# Calculus
This package contains many useful functions for single- and multi-variable calculus.

It also includes scripts to display slope and vector fields, as well as algorithms like Newton's method and Euler's method.

## Contents
**calc_1d.jl** - Single-variable derivative and integral calculator

**calc_2d.jl** - Partial derivatives, double integrals, line integrals, and vector calculus operators, all for 2D.

**calc_3d.jl** - Partial derivatives, triple integrals, line integrals, surface integrals, and vector calculus operators, all for 3D.

**visuals/slope_fields.jl** - Graphing slope fields

**visuals/vector_fields.jl** - Graphing 2D and 3D vector fields

**visuals/newton_method.jl** - Newton's method to solve for zeros of functions

# TensorAlgebra.jl

This Julia package allows for tensor algebra and calculus using the Einstein summation convention and
symbolic indexing. It also includes tools for differential geometry, like the metric tensor, Christoffel
symbols, and covariant derivative.

## Types

Tensor{T, R} - An arbitrary rank R (m, n)-tensor of a type T

KroneckerDelta - The Kronecker Delta δ

LeviCivita - The Levi-Civita Symbol ε

PartialDerivative{N} - The partial derivative operator ∂ on N coordinates

CovariantDerivative - The covariant derivative operator ∇


## Functions
### General

Tensor() - Constructor for Tensor{T, R}

getindex() - Einstein convention indexing

### Algebra

:⊗ - Tensor product

:* - Tensor scaling and contraction

:+ - Tensor addition

:- - Tensor subtraction

:⋅ - Dot product for (1, 0)-tensors

### Geometry

metric() - Metric tensor from a basis

inv() - Invert a (2, 0) or (0, 2)-tensor

christoffel() - Compute the Levi-Civita Connection coefficients

## Examples
Defining a tensor
```julia
julia> L = Tensor([[1, 2]', [3, -1]'])
Tensor{Int64, 2}([1 2; 3 -1], (:contra, :co))
```
Tensor contraction
```julia
julia> v = Tensor([4, -2]); w = Tensor([1, 1]')
julia> v[:i] * w[:i]
2
```
Finding the metric tensor
```julia
julia> v = Tensor([3, 1]); u = Tensor([-1, 2])
julia> metric((u, v))
Tensor{Int64, 2}([5 -1; -1 10], (:co, :co))
```
