using Test
using BenchmarkTools
using polyBindingModel
using ForwardDiff
using LinearAlgebra

"""
multinomial(k...)
Multinomial coefficient where `n = sum(k)`.
"""
function multinomial(k...)
    s = 0
    result = 1
    @inbounds for i in k
        s += i
        result *= binomial(s, i)
    end
    result
end

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

function polyfc_via_polyc(Req, L0::Real, KxStar::Real, f::Number, Rtot::Vector, LigC::Vector, Kav::Matrix)
    LigC /= sum(LigC)
    Cplx = vec_comb(length(LigC), f, repeat([f], length(LigC)))'
    Ctheta = exp.(Cplx * log.(LigC)) .* [multinomial(x...) for x in eachrow(Cplx)]
    @assert sum(Ctheta) ≈ 1.0 "Ctheta is $(Ctheta) with sum $(sum(Ctheta)) != 1.0"

    return polyc_Req(Req, L0, KxStar, Rtot, Cplx, Ctheta, Kav)
end


L0 = 1.0e-8
KxStar = 1.0e-10
Rtot = [100.0, 1000.0, 10.0, 10000.0]
Kav = ones((3, 4)) * 1.0e6
Kav[1, 2] = 1.0e9


@testset "randomBinding tests" begin
    IgGC = [1.0, 0.1, 3.0]

    @testset "Can successfully assemble the parameters and get a sane result." begin
        @btime polyfc($L0, $KxStar, 4, $Rtot, $IgGC, $Kav)

        for ff in 1:20
            out = polyfc(L0, KxStar, ff, Rtot, IgGC, Kav)
            # Mass balance
            @test all(out.Rbound_n .<= Rtot)
            @test all(out.Rbound_n .>= 0.0)
            @test all(out.Req .<= Rtot)
            @test all(isapprox.(out.Rbound_n .+ out.Req, Rtot))

            nres = polyfc_via_polyc(out.Req, L0, KxStar, ff, Rtot, IgGC, Kav)
            @test out.Lbound ≈ sum(nres[1])
            @test isapprox(out.Rbound, sum(nres[2]), rtol = 1e-6)
        end
    end

    @testset "Test monovalent case." begin
        out = polyfc(L0, KxStar, 1, Rtot, [1], Kav[1, :]')

        # Note f is not used
        comp = vec(Kav[1, :]' .* L0 .* Rtot' ./ (1 .+ (Kav[1, :]' .* L0)))

        @test all(out.Lbound .≈ sum(comp))
        @test all(out.Rbound_n .≈ comp)
        @test all(out.Rbound_n .+ out.Req .≈ Rtot)
        @test out.Rmulti == 0.0
        @test all(out.Rmulti_n .== 0.0)
    end
end


@testset "Test Lfbnd in complexBinding when f = 1, 2, 3" begin
    # f = 1
    Cplx = [1 0 0; 0 1 0; 0 0 1]
    θ = rand(3)
    Lbounds, Rbounds, Lfbnds, Lmbnds = polyc(L0, KxStar, Rtot, Cplx, θ / sum(θ), Kav)
    @test all(Lbounds .≈ Lfbnds)
    @test all(Lbounds .≈ sum(Rbounds, dims = 2))
    @test all(Lbounds .≈ Lmbnds)

    # f = 3
    Cplx = [3 0 0; 2 1 0; 2 0 1; 1 2 0; 1 1 1; 1 0 2; 0 3 0; 0 2 1; 0 1 2; 0 0 3]
    θ = rand(10)
    Lbounds, Rbounds, Lfbnds, Lmbnds = polyc(L0, KxStar, Rtot, Cplx, θ / sum(θ), Kav)
    Lbbnds = Lbounds - Lfbnds - Lmbnds
    @test all(sum(Rbounds, dims = 2) .≈ Lfbnds * 3 + Lbbnds * 2 + Lmbnds)
end


include("testForwardDiff.jl")
