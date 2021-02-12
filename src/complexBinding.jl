
function Req_func(Req::Vector, L0::Real, Kx::Real, Rtot::Vector, Cplx::AbstractMatrix, θ::Vector, Kav::Matrix)
    ψ = Kav .* transpose(Req) .* Kx
    ψRS = sum(ψ, dims = 2) .+ 1.0
    ψNorm = ψ ./ ψRS
    Rbound = L0 / Kx * dropdims(sum(θ .* Cplx * ψNorm .* exp.(Cplx * log1p.(ψRS .- 1)), dims = 1), dims = 1)
    return Req .+ Rbound .- Rtot
end


function polyc(L0::Real, KxStar::Real, Rtot::Vector, Cplx::AbstractMatrix, θ::Vector, Kav::Matrix)
    ncplx = size(Cplx, 1)
    @assert size(Cplx, 2) == size(Kav, 1)
    @assert ncplx == length(θ)
    @assert size(Kav, 2) == length(Rtot)
    θ /= sum(θ)

    ansType = promote_type(typeof(L0), typeof(KxStar), eltype(Rtot), eltype(θ))
    f! = (F, x) -> F .= Req_func(x, L0, KxStar, Rtot, Cplx, θ, Kav)
    Req = rootSolve(f!, convert(Vector{ansType}, Rtot))

    Psi = Kav .* transpose(Req) .* KxStar
    PsiRS = sum(Psi, dims = 2) .+ 1.0
    Lbounds = L0 / KxStar * θ .* expm1.(Cplx * log1p.(PsiRS .- 1))
    Rbounds = L0 / KxStar * θ .* Cplx * (Psi ./ PsiRS) .* exp.(Cplx * log1p.(PsiRS .- 1))
    Lfbnds = L0 / KxStar * θ .* exp.(Cplx * log.(PsiRS .- 1.0))
    Lmbnds = L0 / KxStar * θ .* Cplx * (PsiRS .- 1)

    @assert sum(Rbounds) ≈ L0 / KxStar * sum(θ .* (Cplx * (1 .- 1 ./ PsiRS)) .* exp.(Cplx * log.(PsiRS)))
    @assert length(Lbounds) == ncplx
    @assert size(Rbounds, 1) == length(θ)
    @assert size(Rbounds, 2) == length(Rtot)
    return Lbounds, Rbounds, Lfbnds, Lmbnds
end

polycm = (KxStar, Rtot, Cplx, Ltheta, Kav) -> polyc(sum(Ltheta), KxStar, Rtot, Cplx, Ltheta ./ sum(Ltheta), Kav)
