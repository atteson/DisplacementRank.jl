using DisplacementRank
using Random

Random.seed!(1)
u = rand(3)
v = rand(7)

w = DisplacementRank.convolve(u, v)

w2 = zeros(9)
for i = 1:9
    for j = 1:3
        k = i - j + 1
        if 0 < k < 8
            w2[i] += u[j] * v[k]
        end
    end
end

@assert( maximum(abs.(w2 - w)) < 1e-10 )
