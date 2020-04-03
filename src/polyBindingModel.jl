module polyBindingModel

using NLsolve
using LinearAlgebra

include("fcBindingModel.jl")
include("complexBinding.jl")

export polyfc, polyc, polycm, polyfcm

end # module
