
abstract type DisplacementRankMatrix{T <: Number} end

mutable struct ToeplitzMatrix{T} <: DisplacementRankMatrix{T}
    entries::Vector{T}
end

Base.getindex( M::ToeplitzMatrix{T}, i::U, j::U ) where { T, U <: Integer } =
    M.entries[div(length(M.entries)+1,2) + i - j]    
