
function polyc_Req(Req::Vector, L0::Real, Kₓ::Real, Rtot::Vector, Cplx::AbstractMatrix, θ::Vector, Kav::Matrix)
    @assert size(Cplx) == (length(θ), size(Kav, 1))
    @assert size(Kav, 2) == length(Rtot)
    θ /= sum(θ)

    ψ = Kav .* Req' .* Kₓ
    ψRS = sum(ψ, dims = 2) .+ 1.0
    L0Kt = L0 / Kₓ * θ
    Lbounds = L0Kt .* expm1.(Cplx * log.(ψRS))
    Rbounds = L0Kt .* Cplx * (ψ ./ ψRS) .* exp.(Cplx * log.(ψRS))
    Lfbnds = L0Kt .* exp.(Cplx * log.(ψRS .- 1.0))
    Lmbnds = L0Kt .* Cplx * (ψRS .- 1.0)

    @assert length(Lbounds) == size(Cplx, 1)
    @assert size(Rbounds) == (length(θ), length(Rtot))
    return Lbounds, Rbounds, Lfbnds, Lmbnds
end


function polyc(args...)
    ansType = promote_type(typeof(args[1]), typeof(args[2]), eltype(args[3]), eltype(args[4]), eltype(args[5]))
    f! = (F, x) -> F .= args[3] .- vec(sum(polyc_Req(x, args...)[2], dims = 1)) .- x
    Req = rootSolve(f!, convert(Vector{ansType}, args[3]))
    return polyc_Req(Req, args...)
end


polycm = (Kₓ, Rtot, Cplx, Ltheta, Kav) -> polyc(sum(Ltheta), Kₓ, Rtot, Cplx, Ltheta ./ sum(Ltheta), Kav)
