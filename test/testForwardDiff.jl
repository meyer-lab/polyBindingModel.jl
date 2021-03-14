@testset "Test that we can use ForwardDiff" begin
    L0 = 1.0e-8
    KxStar = 1.0e-10
    f = 4
    Rtot = [100.0, 1000.0, 10.0, 10000.0]
    IgGC = [1.0, 0.1, 3.0]
    Kav = ones((3, 4)) * 1.0e6
    Kav[1, 2] = 1.0e9

    @testset "randomBinding" begin
        func = x -> polyfc(L0, KxStar, f, x, IgGC, Kav).Rmulti
        out = ForwardDiff.gradient(func, Rtot)
        @test eltype(out) == Float64
        @test length(out) == length(Rtot)

        func = x -> polyfc(L0, KxStar, x, Rtot, IgGC, Kav).Rmulti
        out = ForwardDiff.derivative(func, Float64(f))
        @test typeof(out) == Float64
        @test out > 0.0

        func = x -> polyfc(x, KxStar, f, Rtot, IgGC, Kav).Rmulti
        out = ForwardDiff.derivative(func, L0)
        @test typeof(out) == Float64

        func = x -> polyfc(L0, x, f, Rtot, IgGC, Kav).Rmulti
        out = ForwardDiff.derivative(func, KxStar)
        @test typeof(out) == Float64
    end

    Cplx = [1 0 3; 2 2 0; 1 1 2; 4 0 0]
    Ctheta = [0.1, 0.3, 0.5, 0.1]
    Ltheta = L0 .* Ctheta

    @testset "complexBinding" begin
        for i = 1:4
            func = x -> sum(polyc(x, KxStar, Rtot, Cplx, Ctheta, Kav)[i])
            out = ForwardDiff.derivative(func, L0)
            @test typeof(out) == Float64

            func = x -> sum(polyc(L0, x, Rtot, Cplx, Ctheta, Kav)[i])
            out = ForwardDiff.derivative(func, KxStar)
            @test typeof(out) == Float64

            func = x -> sum(polyc(L0, KxStar, x, Cplx, Ctheta, Kav)[i])
            out = ForwardDiff.gradient(func, Rtot)
            @test eltype(out) == Float64
            @test length(out) == length(Rtot)

            func = x -> sum(polyc(L0, KxStar, Rtot, Cplx, x, Kav)[i])
            out = ForwardDiff.gradient(func, Ctheta)
            @test eltype(out) == Float64
            @test length(out) == length(Ctheta)

            func = x -> sum(polycm(KxStar, Rtot, Cplx, x, Kav)[i])
            out = ForwardDiff.gradient(func, Ltheta)
            @test eltype(out) == Float64
            @test length(out) == length(Ltheta)
        end
    end
end
