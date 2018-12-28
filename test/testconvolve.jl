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

u = [rand(Complex{Float64}, 16); zeros(Complex{Float64}, 16)]
v = [rand(Complex{Float64}, 16); zeros(Complex{Float64}, 16)]

w = zeros(Complex{Float64}, 32)
for i = 1:32
    for j = 1:32
        k = i - j + 1
        if 0 < k < 33
            w[i] += u[j] * v[k]
        end
    end
end

w2 = zeros(Complex{Float64}, 32)
u2 = copy(u)
v2 = copy(v)    
@time DisplacementRank.convolve!( u2, v2, w2 )
@assert( maximum(abs.(w - w2)) .< 1e-12 )

w2 = zeros(Complex{Float64}, 32)
u2 = copy(u)
v2 = copy(v)    
@time DisplacementRank.convolve!( u2, v2, w2 )
@assert( maximum(abs.(w - w2)) .< 1e-12 )

u = rand(10000)
v = rand(10000)
@time DisplacementRank.convolve(u, v)

using FFTW
u = rand(Complex{Float64}, 2^20)
u0 = copy(u)
plan = plan_fft( u )
u2 = copy(u)
using LinearAlgebra
@time mul!( u2, plan, u );

f = rand(10)
