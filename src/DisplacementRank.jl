module DisplacementRank

using FFTW

abstract type DisplacementRankMatrix{T <: Number} end

mutable struct ToeplitzMatrix{T} <: DisplacementRankMatrix{T}
    entries::Vector{T}
end

Base.getindex( M::ToeplitzMatrix{T}, i::U, j::U ) where { T, U <: Integer } =
    M.entries[div(length(M.entries)+1,2) + i - j]

function convolve( u::AbstractVector{T}, v::AbstractVector{T} ) where {T <: Number}
    n = length(u)
    m = length(v)
    h = n + m - 1
    N = 2^Int(ceil(log2(h)))
    z = zero(T)
    
    resize!( u, N )
    for i = n+1:N
        u[i] = z
    end
    
    resize!( v, N )
    for i = m+1:N
        v[i] = z
    end

    fu = rfft(u)
    fv = rfft(v)

    w = irfft(fu .* fv, N)
    return w[1:h]
end

function convolve!( u::AbstractVector{Complex{T}},
                   v::AbstractVector{Complex{T}},
                   w::AbstractVector{Complex{T}} ) where {T <: Number}
    fft!( u )
    fft!( v )
    w[1:32] = u .* v
    ifft!( w )
end

end # module
