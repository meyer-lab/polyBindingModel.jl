module polyBindingModel

using NLsolve


function rootSolve(f!, Rtot)
    local solve_res
    try
        solve_res = nlsolve(f!, Rtot, method = :newton, autodiff = :forward, ftol = 1e-9, xtol = 1e-10)
        @assert solve_res.f_converged == true
        @assert all(solve_res.zero .<= Rtot .+ eps())
        @assert all(-eps() .<= solve_res.zero)
    catch e
        println("Req solving failed")
        rethrow(e)
    end

    return solve_res.zero
end


include("randomBinding.jl")
include("complexBinding.jl")

export polyfc, polyc, polycm, polyfcm

end # module
