import Combinatorics.multinomial

function vec_comb(length, rsum, resid)
    if length <= 1
        return fill(rsum, (1, 1))
    end
    enum = Array{Int64}(undef, length, 0)
    for i = max(0, rsum - sum(resid[2:end])):min(resid[1], rsum)
        other_ele = vec_comb(length - 1, rsum - i, resid[2:end])
        first_ele = ones(Int64, 1, size(other_ele, 2)) * i
        enum = hcat(enum, vcat(first_ele, other_ele))
    end
    return enum
end

function polyfc_via_polyc(L0::Real, KxStar::Real, f::Number, Rtot::Vector, LigC::Vector, Kav::Matrix)
    LigC /= sum(LigC)
    Cplx = vec_comb(length(LigC), f, repeat([f], length(LigC)))'
    Ctheta = exp.(Cplx * log.(LigC)) .* [multinomial(x...) for x in eachrow(Cplx)]
    @assert sum(Ctheta) ≈ 1.0 "Ctheta is $(Ctheta) with sum $(sum(Ctheta)) != 1.0"

    return polyc(L0, KxStar, Rtot, Cplx, Ctheta, Kav)
end


@testset "Give the same results as fcBindingModel" begin
    for i = 1:10
        L0 = rand() * 10.0^rand(-15:-5)
        KxStar = rand() * 10.0^rand(-15:-5)
        f = rand(2:6)
        nl = rand(1:6)
        nr = rand(1:6)

        Rtot = floor.(100 .+ rand(nr) .* (10 .^ rand(4:6, nr)))
        LigC = rand(nl) .* (10 .^ rand(1:2, nl))
        Kav = rand(nl, nr) .* (10 .^ rand(3:7, nl, nr))

        old_res = polyfc(L0, KxStar, f, Rtot, LigC, Kav)
        new_res = polyfc_via_polyc(L0, KxStar, f, Rtot, LigC, Kav)

        @test old_res.Lbound ≈ new_res[1]
        @test old_res.Rbound ≈ new_res[2]
    end
end

@testset "complexBinding can take ForwardDiff" begin
    L0 = 1.0e-9
    KxStar = 1.2e-10
    Rtot = [100.0, 1000.0, 10.0, 10000.0]
    Cplx = [1 0 3; 2 2 0; 1 1 2; 4 0 0]
    Ctheta = rand(4)
    Ctheta = Ctheta / sum(Ctheta)
    Kav = rand(3, 4) * 1.0e7
    Ltheta = L0 .* Ctheta

    for i = 1:2
        func = x -> polyc(x, KxStar, Rtot, Cplx, Ctheta, Kav)[i]
        out = ForwardDiff.derivative(func, L0)
        @test typeof(out) == Float64

        func = x -> polyc(L0, x, Rtot, Cplx, Ctheta, Kav)[i]
        out = ForwardDiff.derivative(func, KxStar)
        @test typeof(out) == Float64

        func = x -> polyc(L0, KxStar, x, Cplx, Ctheta, Kav)[i]
        out = ForwardDiff.gradient(func, Rtot)
        @test eltype(out) == Float64
        @test length(out) == length(Rtot)

        func = x -> polyc(L0, KxStar, Rtot, Cplx, x, Kav)[i]
        out = ForwardDiff.gradient(func, Ctheta)
        @test eltype(out) == Float64
        @test length(out) == length(Ctheta)

        func = x -> polycm(KxStar, Rtot, Cplx, x, Kav)[i]
        out = ForwardDiff.gradient(func, Ltheta)
        @test eltype(out) == Float64
        @test length(out) == length(Ltheta)
    end
end
