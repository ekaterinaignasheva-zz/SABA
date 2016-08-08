# SABA
Master thesis (2016, simplectic integration)

Scientific director is Lapteva T.V. (Lobachevsky University of Nighny Novgorod)

In scope of this thesis I have implemented high accurate numerical integration scheme.
This type of intergation requires big amount of floating point operations. 
Main achievement is an obtained performance gain that is about ~10e4 times decrease.
CUDA API was used to provide such high results.
See all particular information about purpose of thesis, implementation steps and HW/SW configuration in the doumentation inside.

Two main theory sources are: 

1. Nonlinear lattice waves in heterogeneous media. T. V. Laptyeva, M. V. Ivanchenko and S.Flach. Topical Review, J. Phys. A: Math. Theor. 47, 493001, 2014.  

2. J. Froihlich, T. Spencer and C. E. Wayne. Localization in disordered, nonlinear dynamical systems. J. Stat. Phys., vol. 42, page 247, 1986.

You can find:

* thesis text in ./SABA/Docs/Dissertation (in Russian only),

* presentation slides in ./SABA/Docs/Presentation (in Russian only),

* code in ./SABA/SABA/kernel,

* description of programm output in ./SABA/Docs/Descriptions.
