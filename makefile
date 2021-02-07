all: figures

figures:
	julia -e 'include("figures/figures.jl")'

clean:
	rm -rf *.svg