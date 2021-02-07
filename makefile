all: figure

figure:
	julia -e 'include("figures/figures.jl")'

clean:
	rm -rf *.svg