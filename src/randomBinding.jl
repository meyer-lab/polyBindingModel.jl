
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
    (ni, nr) = size(Kav)
    @assert ni == length(IgGC)
    @assert nr == length(Rtot)
    IgGC /= sum(IgGC)

    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC))

    Av = transpose(Kav) * IgGC * Kₓ
    f! = (F, x) -> F .= x + L0 * f / Kₓ .* (x .* Av) * (1 + sum(x .* Av))^(f - 1) - Rtot

    Req = rootSolve(f!, convert(Vector{ansType}, Rtot))

    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC))
    Phi = ones(ansType, ni, nr + 1) .* IgGC
    Phi[:, 1:nr] .*= Kav .* transpose(Req) .* Kₓ
    Phisum = sum(Phi[:, 1:nr])
    Phisum_n = sum(Phi[:, 1:nr], dims = 1)

    w = fcOutput{ansType}(
        L0 / Kₓ * ((1 + Phisum)^f - 1),
        L0 / Kₓ * f * Phisum * (1 + Phisum)^(f - 1),
        L0 / Kₓ * f * Phisum * ((1 + Phisum)^(f - 1) - 1),
        Req,
        vec(L0 / Kₓ * f .* Phisum_n * (1 + Phisum)^(f - 1)),
        vec(L0 / Kₓ * f .* Phisum_n * ((1 + Phisum)^(f - 1) - 1)),
    )
    return w
end

polyfcm = (Kₓ, f, Rtot, IgG, Kav) -> polyfc(sum(IgG) / f, Kₓ, f, Rtot, IgG ./ sum(IgG), Kav)
