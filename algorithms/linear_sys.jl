using LinearAlgebra

M = [1 1; 2 0]

function find_soln(M, v₀)
    evecs = eigvecs(M)
    evals = eigvals(M)
    coef = inv(evecs) * v₀

    function f(t)
        linear_comb = zeros(2)
        for i in 1:length(M[1, begin:end])
            # Solutions look like a linear combination of c*exp(λt)*v
            linear_comb += coef[i] * exp(evals[i] * t) * evecs[i, begin:end]
        end
        return linear_comb
    end

    return f
end

f = find_soln(M, [1, 1])

f(3)