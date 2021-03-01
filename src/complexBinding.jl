
function polyc_Req(Req::Vector, L0::Real, Kₓ::Real, Rtot::Vector, Cplx::AbstractMatrix, θ::Vector, Kav::Matrix)
    @assert size(Cplx) == (length(θ), size(Kav, 1))
    @assert size(Kav, 2) == length(Rtot)
    θ /= sum(θ)

    Psi = Kav .* Req' .* Kₓ
    PsiRS = sum(Psi, dims = 2) .+ 1.0
    Lbounds = L0 / Kₓ * θ .* expm1.(Cplx * log1p.(PsiRS .- 1))
    Rbounds = L0 / Kₓ * θ .* Cplx * (Psi ./ PsiRS) .* exp.(Cplx * log1p.(PsiRS .- 1))
    Lfbnds = L0 / Kₓ * θ .* exp.(Cplx * log.(PsiRS .- 1.0))
    Lmbnds = L0 / Kₓ * θ .* Cplx * (PsiRS .- 1)

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
