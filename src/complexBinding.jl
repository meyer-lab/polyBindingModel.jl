
function Req_func(Req::Vector, L0::Real, KxStar::Real, Rtot::Vector, Cplx::AbstractMatrix, Ctheta::Vector, Kav::Matrix)
    Psi = Kav .* transpose(Req) .* KxStar
    PsiRS = sum(Psi, dims = 2) .+ 1.0
    PsiNorm = (Psi ./ PsiRS)
    Rbound = L0 / KxStar * dropdims(sum(Ctheta .* Cplx * PsiNorm .* exp.(Cplx * log1p.(PsiRS .- 1)), dims = 1), dims = 1)
    return Req .+ Rbound .- Rtot
end


function polyc(L0::Real, KxStar::Real, Rtot::Vector, Cplx::AbstractMatrix, Ctheta::Vector, Kav::Matrix)
    ncplx = size(Cplx, 1)
    @assert size(Cplx, 2) == size(Kav, 1)
    @assert ncplx == length(Ctheta)
    @assert size(Kav, 2) == length(Rtot)
    Ctheta /= sum(Ctheta)

    ansType = promote_type(typeof(L0), typeof(KxStar), eltype(Rtot), eltype(Ctheta))
    f! = (F, x) -> F .= Req_func(x, L0, KxStar, Rtot, Cplx, Ctheta, Kav)
    Req = rootSolve(f!, convert(Vector{ansType}, Rtot))

    Psi = Kav .* transpose(Req) .* KxStar
    PsiRS = sum(Psi, dims = 2) .+ 1.0
    Lbounds = L0 / KxStar * Ctheta .* expm1.(Cplx * log1p.(PsiRS .- 1))
    Rbounds = L0 / KxStar * dropdims(sum(Ctheta .* Cplx * (Psi ./ PsiRS) .* exp.(Cplx * log1p.(PsiRS .- 1)), dims = 1), dims = 1)
    Lfbnds = L0 / KxStar * Ctheta .* exp.(Cplx * log.(PsiRS .- 1.0))
    Lmbnds = L0 / KxStar * Ctheta .* Cplx * (PsiRS .- 1)

    @assert sum(Rbounds) ≈ L0 / KxStar * sum(Ctheta .* (Cplx * (1 .- 1 ./ PsiRS)) .* exp.(Cplx * log.(PsiRS)))
    @assert length(Lbounds) == ncplx
    @assert length(Rbounds) == length(Rtot)
    return Lbounds, Rbounds, Lfbnds, Lmbnds
end

polycm = (KxStar, Rtot, Cplx, Ltheta, Kav) -> polyc(sum(Ltheta), KxStar, Rtot, Cplx, Ltheta ./ sum(Ltheta), Kav)
