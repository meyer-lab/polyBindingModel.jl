using Roots

struct fcOutput{T}
    Lbound::T
    Rbound::T
    Rmulti::T
    Req::Vector{T}
    Rbound_n::Vector{T}
    Rmulti_n::Vector{T}
end


function phi_func(ϕs::Real, Rtot::Vector, L0::Real, Kₓ::Real, f::Number, A::Vector)
    Req = Rtot ./ (1.0 .+ L0 * f * A * (1 + ϕs) ^ (f - 1))
    return ϕs - Kₓ * dot(A, Req)
end


function polyfc(L0::Real, Kₓ::Real, f::Number, Rtot::Vector, IgGC::Vector, Kav::AbstractMatrix)
    # Data consistency check
    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC))
    @assert size(Kav) == (length(IgGC), length(Rtot))
    @assert ndims(Kav) == 2
    IgGC /= sum(IgGC)

    # Setup constant terms
    if length(IgGC) > 1
        A = vec(Kav' * IgGC)
    else
        A = vec(Kav)
    end
    L0fK = L0 * f / Kₓ

    high = Kₓ * dot(A, Rtot)
    func = x -> phi_func(x, Rtot, L0, Kₓ, f, A)
    Phisum = find_zero(func, (convert(ansType, 0.0), convert(ansType, high)), Bisection())

    Req = Rtot ./ (1.0 .+ L0 * f * A * (1 + Phisum) ^ (f - 1))
    Phi = ones(ansType, size(Kav, 1), size(Kav, 2) + 1) .* IgGC
    Phi[:, 1:size(Kav, 2)] .*= Kav .* Req' .* Kₓ
    Phisum_n = sum(Phi[:, 1:size(Kav, 2)], dims = 1)

    w = fcOutput{ansType}(
        L0 / Kₓ * ((1 + Phisum)^f - 1),
        sum(Rtot - Req),
        L0fK * Phisum * ((1 + Phisum)^(f - 1) - 1),
        Req,
        vec(L0fK .* Phisum_n * (1 + Phisum)^(f - 1)),
        vec(L0fK .* Phisum_n * ((1 + Phisum)^(f - 1) - 1)),
    )
    return w
end


polyfcm = (Kₓ, f, Rtot, IgG, Kav) -> polyfc(sum(IgG) / f, Kₓ, f, Rtot, IgG ./ sum(IgG), Kav)
