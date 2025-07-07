function gram_schmidt(B, ip)
    C = []
    for i in eachindex(B)
        vᵢ = B[i]
        w = B[i]
        for j in eachindex(B)
            uⱼ = B[j]
            if i != j
                w -= (ip(vᵢ, uⱼ) / ip(uⱼ, uⱼ)) * uⱼ
            end
        end
        mag = ip(w, w)
        append!(C, [w / sqrt(mag)])
    end

    return C
end

function norm(v, ip)
    return sqrt(ip(v, v))
end

B = ([1, 1, 1], [1, 1, 0], [1, 0, 0])
ip(v, w) = (v[1] * w[1]) + (v[2] * w[2]) + (v[3] * w[3])

C = gram_schmidt(B, ip)
println(C)