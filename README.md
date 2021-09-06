# Magnetohydrodynamics2D
Incompressible MHD in two dimensions with mixed discontinuous Galerkin / Fourier spectral method

<p align="center">
<img src="https://raw.githubusercontent.com/crewsdw/Magnetohydrodynamics2D/master/images/k234_rs150/vorticity.png" width="400" />
<img src="https://raw.githubusercontent.com/crewsdw/Magnetohydrodynamics2D/master/images/k234_rs150/current.png" width="400" />
</p>
  
Variables: System solved in Elsasser formulation in 2D. Velocity and magnetic fields are in-plane.

Work in progress utilizing experimental spectral techniques

Project objectives mirror Incompressible2D:
1) concisely-coded and efficient GPU implementations,
2) some experimental spectral methods for the pressure Poisson equation and viscosity

Implementation notes can be found at: https://students.washington.edu/dcrews/notes/mhd_poisson.pdf
