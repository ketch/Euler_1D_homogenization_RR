#!/usr/bin/env python
# encoding: utf-8
r"""
NOTE: In this version we initialize using Eulerian coordinates.


Euler solitary wave formation
=============================

Solve the one-dimensional compressible Euler equations:

.. math::
    \rho_t + (\rho u)_x & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x & = 0 \\
    E_t + (u (E + p) )_x & = 0.

The initial condition corresponds to a setup that generates
a train of solitary waves.  The background density is varying sinusoidally
while the pressure is constant.  The initial velocity is zero.
"""
import numpy as np
from clawpack import riemann
from clawpack.riemann.euler_with_efix_1D_constants import density, momentum, energy, num_eqn
from scipy.integrate import quad
from scipy.interpolate import interp1d

gamma = 1.4  # Ratio of specific heats

def setup(mx=2400, xright=600., tfinal=600., nout=20,
          use_petsc=False,iplot=False,htmlplot=False,outdir='./_output',
          solver_type='classic',periodic=True,entropy_field=None,
          entropy_amp=0.8):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    rs = riemann.euler_with_efix_1D
    if solver_type == "classic":
        solver = pyclaw.ClawSolver1D(rs)
    elif solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver1D(rs)

    solver.bc_lower[0]=pyclaw.BC.periodic
    solver.bc_upper[0]=pyclaw.BC.periodic

    x = pyclaw.Dimension(-xright,xright,mx,name='x')
    dx = x.delta
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma']= gamma

    xc = state.grid.p_centers[0]

    pressure = 1+0.15*np.exp(-xc**2/16);
    velocity = xc * 0.

    if use_petsc:
        owned_cells = domain.patch._da.getRanges()[0]
    else:
        owned_cells = [0,mx]


    if entropy_field is None:
        state.q[density ,:] = 1 + entropy_amp*np.cos(2*np.pi*xc)
    else:
        prescribed_field = np.loadtxt(entropy_field)
        state.q[density ,:] = prescribed_field[owned_cells[0]:owned_cells[1]]

    state.q[momentum,:] = 0.
    state.q[energy  ,:] = pressure/(gamma - 1.) + 0.5 * state.q[density,:] * velocity**2

    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = nout
    claw.outdir = outdir
    if mx < 100000:
        claw.keep_copy = True
    else:
        claw.keep_copy = False

    return claw


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
