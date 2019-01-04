using FFTW
using LinearAlgebra
using Random
using DisplacementRank
using PyPlot
using Brobdingnag

n = 10000
N = 2*n-1

#Random.seed!(1)
#x = rand(N)
#x = x ./ (((1:N) .- n).^2 .+ 1)
x = [zeros(n-2);0.5;1.0;0.5;zeros(n-2)];
f = ifft(x)*length(x);

f = exp.(pi*im*(N+1)/N * (0:N-1)) .* f;


X = [x[n + i - j] for i in 1:n, j in 1:n];
logdet(X)
sum(log.(f))

struct Brob
    positive::Bool
    log::Float64
end

Brob( x::Float64 ) = Brob( sign(x) > 0, log(abs(x)) )

convert( Float64, x::Brob) = (x.positive ? 1 : -1 )*exp( x.log )

function Base.:+( x::Brob, y::Brob )
    sx = x.positive
    sy = y.positive
    if sx == sy
        return Brob( sx, x.log + log(1 + exp(y.log - x.log)) )
    else
        if x.log > y.log
            return Brob( sx, x.log + log(1 - exp(y.log - x.log)) )
        else
            return Brob( sy, y.log + log(1 - exp(x.log - y.log)) )
        end
    end
end

Base.:-( x::Brob, y::Brob ) = x + Brob(!y.positive, y.log)

Base.:*( x::Brob, y::Brob ) = Brob( x.positive == y.positive, x.log + y.log )

Base.:/( x::Brob, y::Brob ) = Brob( x.positive == y.positive, x.log - y.log )

function Base.:^( x::Brob, n::Int )
    @assert( x.positive )
    return Brob( true, n*x.log )
end


function inverse( a, m )
    n = length(a)
    b = [Brob(1/a[1])]
    for i = 2:m
        push!( b, Brob(-a[2]/a[1])*b[i-1] )
        for j = 3:min(i,n)
            b[i] = b[i] + Brob(-a[j]/a[1]) * b[i - j + 1]
        end
    end
    return b
end

LinearAlgebra.det( a, n ) = Brob((-1.0)^n)*Brob(a[1])^(n+1)*inverse( a, n+1 )[n+1]

a = [0.5,1.0,0.5]
@time d = det( a, N )
sum(convert( Float64, d )
@time LinearAlgebra.logdet([abs(i - j) > 1 ? 0.0 : a[i-j+2] for i in 1:n, j in 1:n])

det( a, 200 )
