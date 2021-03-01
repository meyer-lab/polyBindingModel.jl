
mutable struct fcOutput{T}
    Lbound::T
    Rbound::T
    Rmulti::T
    Req::Vector{T}
    Rbound_n::Vector{T}
    Rmulti_n::Vector{T}
end


function polyfc(L0::Real, Kₓ::Real, f::Number, Rtot::Vector, IgGC::Vector, Kav::AbstractMatrix)
    # Data consistency check
    @assert size(Kav, 1) == length(IgGC)
    @assert size(Kav, 2) == length(Rtot)
    IgGC /= sum(IgGC)

    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC))

    Av = transpose(Kav) * IgGC * Kₓ
    L0fK = L0 * f / Kₓ
    f! = (F, x) -> F .= x + L0fK .* (x .* Av) * (1 + x' * Av)^(f - 1) - Rtot

    Req = rootSolve(f!, convert(Vector{ansType}, Rtot))

    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC))
    Phi = ones(ansType, size(Kav, 1), size(Kav, 2) + 1) .* IgGC
    Phi[:, 1:size(Kav, 2)] .*= Kav .* transpose(Req) .* Kₓ
    Phisum_n = sum(Phi[:, 1:size(Kav, 2)], dims = 1)
    Phisum = sum(Phisum_n)

    w = fcOutput{ansType}(
        L0 / Kₓ * ((1 + Phisum)^f - 1),
        sum(Rtot - Req),
        L0fK * Phisum * ((1 + Phisum)^(f - 1) - 1),
        Req,
        Rtot - Req,
        vec(L0fK .* Phisum_n * ((1 + Phisum)^(f - 1) - 1)),
    )
    return w
end

polyfcm = (Kₓ, f, Rtot, IgG, Kav) -> polyfc(sum(IgG) / f, Kₓ, f, Rtot, IgG ./ sum(IgG), Kav)
