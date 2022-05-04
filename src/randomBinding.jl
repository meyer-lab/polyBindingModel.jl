using Roots

struct fcOutput{T}
    Lbound::T
    Rbound::T
    Rmulti::T
    Req::Vector{T}
    Rbound_n::Vector{T}
    Rmulti_n::Vector{T}
end


function polyfc(L0::Real, Kₓ::Real, f::Number, Rtot::Vector, IgGC::Vector, Kav::AbstractMatrix)
    # Data consistency check
    ansType = promote_type(typeof(L0), typeof(Kₓ), typeof(f), eltype(Rtot), eltype(IgGC), eltype(Kav))
    @assert size(Kav) == (length(IgGC), length(Rtot))
    @assert ndims(Kav) == 2

    # More sanity checks
    @assert 0.0 <= L0 < Inf "Can't have L0 = $L0"
    @assert 0.0 <= Kₓ < Inf "Can't have Kₓ = $Kₓ"
    @assert 0.0 < f < Inf "Can't have f = $f"
    @assert all(0.0 .<= Rtot .< Inf) "Can't have Rtot = $Rtot"
    @assert all(0.0 .<= IgGC .< Inf) "Can't have IgGC = $IgGC"
    @assert all(0.0 .<= Kav .< Inf) "Can't have Kav = $Kav"
    
    if sum(IgGC) > 0.0
        IgGC /= sum(IgGC)
    else
        L0 = 0.0
        IgGC = ones(length(IgGC)) / length(IgGC)
    end

    # Setup constant terms
    if length(IgGC) > 1
        A = vec(Kav' * IgGC)
    else
        A = vec(Kav)
    end
    L0fK = L0 * f / Kₓ
    L0fA = L0 * f * A

    # Solve for Phisum
    function phi_func(ϕs::Real)
        return ϕs - Kₓ * dot(A, Rtot ./ (1.0 .+ L0fA .* (1 .+ ϕs) .^ (f .- 1)))
    end

    fx = ZeroProblem(phi_func, convert(ansType, 0.0))
    Phisum = solve(fx, Roots.Order1(), atol=0.0, rtol=0.0)
    @assert Phisum >= 0.0
    @assert Phisum <= Kₓ * dot(A, Rtot)

    Req = Rtot ./ (1.0 .+ L0fA * (1 + Phisum) ^ (f - 1))
    Phisum_n = sum(ones(ansType, size(Kav, 1), size(Kav, 2)) .* IgGC .* Kav .* Req' .* Kₓ, dims = 1)

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
