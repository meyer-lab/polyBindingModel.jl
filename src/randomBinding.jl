
mutable struct fcOutput{T}
    Lbound::T
    Rbound::T
    Rmulti::T
    Req::Vector{T}
    Rbound_n::Vector{T}
    Rmulti_n::Vector{T}
end


function polyfc_Req(Req::Vector, L0::Real, Kₓ::Real, f::Number, Rtot::Vector, IgGC::Vector, Kav::AbstractMatrix)
    # Data consistency check
    @assert size(Kav, 1) == length(IgGC)
    @assert size(Kav, 2) == length(Rtot)
    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC), eltype(Req))

    IgGC /= sum(IgGC)
    L0fK = L0 * f / Kₓ

    Phi = ones(ansType, size(Kav, 1), size(Kav, 2) + 1) .* IgGC
    Phi[:, 1:size(Kav, 2)] .*= Kav .* transpose(Req) .* Kₓ
    Phisum_n = sum(Phi[:, 1:size(Kav, 2)], dims = 1)
    Phisum = sum(Phisum_n)

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


function polyfc(args...)
    ansType = promote_type(typeof(args[1]), typeof(args[2]), typeof(args[3]), eltype(args[4]), eltype(args[5]))
    f! = (F, x) -> F .= args[4] .- polyfc_Req(x, args...).Rbound_n .- x
    Req = rootSolve(f!, convert(Vector{ansType}, args[4]))
    return polyfc_Req(Req, args...)
end


polyfcm = (Kₓ, f, Rtot, IgG, Kav) -> polyfc(sum(IgG) / f, Kₓ, f, Rtot, IgG ./ sum(IgG), Kav)
