#!/usr/bin/env python
# encoding: utf-8
r"""
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
pstar = 1.
Vstar = 1.
width = 5.
alpha = 1./8.
SA = -gamma*np.log(1./4)
SB = -gamma*np.log(7./4)

def setup(mx=2400, x0=0., xright=600., tfinal=600., nout=20,
          use_petsc=False,iplot=False,htmlplot=False,outdir='./_output',
          solver_type='classic',periodic=False,variation="pwc",amp=0.15,
          gauges=False):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    rs = riemann.euler_with_efix_1D
    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
    else:
        solver = pyclaw.ClawSolver1D(rs)

    solver.max_steps = 1000000

    if periodic:
        solver.bc_lower[0]=pyclaw.BC.periodic
        solver.bc_upper[0]=pyclaw.BC.periodic
    else:
        solver.bc_lower[0]=pyclaw.BC.wall
        solver.bc_upper[0]=pyclaw.BC.extrap

    mx = mx;
    x = pyclaw.Dimension(0.,xright,mx,name='x')
    dx = x.delta
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    if use_petsc:
        owned_cells = domain.patch._da.getRanges()[0]
    else:
        owned_cells = [0,mx]

    state.problem_data['gamma']= gamma

    # ====================
    # Changing coordinates
    # ====================
    p0 = lambda xi: pstar+amp*np.exp(-((xi-x0)/width)**2.) # Initial pressure
    if variation == "none":
        S = lambda xi: 1/2.
    elif variation == "smooth":
        S = lambda xi: (SA+SB)/2. + (SA-SB)/2. * np.sin(2*np.pi*xi)
    else: # piecewise-constant entropy
        S = lambda xi: SA*((xi-np.floor(xi))<alpha)+SB*((xi-np.floor(xi))>=alpha) # Initial entropy

    V0 = lambda xi: (p0(xi)*np.exp(-S(xi)))**(-1./gamma) # Initial specific volume
    rho0_xi = lambda xi: 1/V0(xi)  # Initial density

    xii = 0
    rho = np.zeros_like(x.centers)
    pressure = np.zeros_like(x.centers)
    for i in range(mx):
        rho[i] = rho0_xi(xii)
        pressure[i] = p0(xii)
        xii = xii + dx*rho[i]
    # ====================

    xc = state.grid.p_centers[0]

    velocity = xc * 0.

    state.q[density ,:] = rho[owned_cells[0]:owned_cells[1]]
    state.q[momentum,:] = velocity * state.q[density,:]
    state.q[energy  ,:] = pressure[owned_cells[0]:owned_cells[1]]/(gamma - 1.) + 0.5 * state.q[density,:] * velocity**2

    if gauges:
        state.grid.add_gauges([(150.,)])
        state.grid.add_gauges([(250.,)])
        state.grid.add_gauges([(350.,)])
        state.grid.add_gauges([(450.,)])
        state.grid.add_gauges([(550.,)])

        state.grid.add_gauges([(650.,)])
        state.grid.add_gauges([(650.25,)])
        state.grid.add_gauges([(650.5,)])
        state.grid.add_gauges([(650.75,)])
        state.grid.add_gauges([(651.,)])
        state.grid.add_gauges([(660.,)])
        state.grid.add_gauges([(670.,)])
        state.grid.add_gauges([(680.,)])

    claw = pyclaw.Controller()
    claw.tfinal = tfinal
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = nout
    claw.outdir = outdir
    claw.keep_copy = True

    return claw


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
