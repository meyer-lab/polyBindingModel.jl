module polyBindingModel

using NLsolve
using LinearAlgebra


function rootSolve(f!, Rtot)
    local solve_res
    try
        solve_res = nlsolve(f!, Rtot * 0.9, method = :newton, autodiff = :forward, iterations = 5000)
        @assert solve_res.f_converged == true
        @assert all(solve_res.zero .<= Rtot .+ eps())
        @assert all(-eps() .<= solve_res.zero)
    catch e
        println("Req solving failed")
        rethrow(e)
    end

    return solve_res.zero
end


include("fcBindingModel.jl")
include("complexBinding.jl")

export polyfc, polyc, polycm, polyfcm

end # module
