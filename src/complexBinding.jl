
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
    Lbound = L0 / KxStar * sum(Ctheta .* expm1.(Cplx * log1p.(PsiRS .- 1)))
    Rbound = L0 / KxStar * sum(Ctheta .* (Cplx * (1 .- 1 ./ PsiRS)) .* exp.(Cplx * log.(PsiRS)))
    Lfbnd = L0 / KxStar * sum(Ctheta .* exp.(Cplx * log.(PsiRS .- 1.0)))
    return Lbound, Rbound, Lfbnd
end

polycm = (KxStar, Rtot, Cplx, Ltheta, Kav) -> polyc(sum(Ltheta), KxStar, Rtot, Cplx, Ltheta ./ sum(Ltheta), Kav)
