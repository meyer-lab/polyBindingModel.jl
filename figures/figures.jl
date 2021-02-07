using Pkg
Pkg.activate(".")
Pkg.add("Gadfly")
Pkg.add("Compose")
Pkg.instantiate()

using polyBindingModel
using Gadfly
using Compose
X = rand(10)
Y = rand(10)
draw(SVG("figure1.svg", 1000px, 1000px), plot(X, x=X, y=Y))