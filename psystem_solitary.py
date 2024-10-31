#!/usr/bin/env python
# encoding: utf-8
r"""
Solitary wave formation in Lagrangian gas dynamics
===========================================================

Solve the one-dimensional p-system:

.. math::
    V_t - u_x & = 0 \\
    u_t + p(V,s(x))_x & = 0.

Here `V` is the specific volume, `p` is the pressure, `s(x)` is the entropy,
and u is the velocity.
We take the equation of state :math:`p = p_* e^{S(x)} (V_*/V)^\gamma`.

An initial hump evolves into two trains of solitary waves.

"""
import numpy as np

gamma = 1.4
#SA    = 1.#np.log(1.0)*gamma/(gamma-1)
#SB    = 4.#np.log(16.0)*gamma/(gamma-1)
rhoA = 1./4
rhoB = 7./4
SA = -gamma*np.log(rhoA)
SB = -gamma*np.log(rhoB)
alpha = rhoA/(rhoA+rhoB)
print(SA,SB)
Vstar=1.
pstar=1.
 
def qinit(state,amp=1.0,xupper=600.,width=5):
    x = state.grid.x.centers
    S = state.aux[0,:]
    
    # Gaussian
    p = pstar+amp*np.exp(-(x/width)**2.)
    state.q[0,:] = Vstar*(p/pstar * np.exp(-S))**(-1./state.problem_data['gamma'])
    state.q[1,:] = 0.


def setaux(x,SB=4,SA=1,alpha=0.5,xlower=0.,xupper=600.,bc=2):
    aux = np.empty([1,len(x)],order='F')
    xfrac = x-np.floor(x)
    # Entropy
    aux[0,:] = SA*(xfrac<alpha)+SB*(xfrac>=alpha)

    # Sinusoidal background
    #aux[0,:] = 4+3.5*np.sin(2*np.pi*x)
    #aux[1,:] = 4+3.5*np.cos(2*np.pi*x)
    return aux

    
def b4step(solver,state):
    #Reverse velocity at trtime
    #Note that trtime should be an output point
    if state.t>=state.problem_data['trtime']-1.e-10 and not state.problem_data['trdone']:
        state.q[1,:]=-state.q[1,:]
        state.q=state.q
        state.problem_data['trdone']=True
        if state.t>state.problem_data['trtime']:
            print('WARNING: trtime is '+str(state.problem_data['trtime'])+\
                ' but velocities reversed at time '+str(state.t))
    #Change to periodic BCs after initial pulse 
    if state.t>5*state.problem_data['tw1'] and solver.bc_lower[0]==0:
        solver.bc_lower[0]=2
        solver.bc_upper[0]=2
        solver.aux_bc_lower[0]=2
        solver.aux_bc_upper[0]=2


def setup(use_petsc=0,solver_type='classic',outdir='./_output',
            tfinal=500.,nout=100, cells_per_layer=40, amp=0.15, L=400.):
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    rs = riemann.psystem_fwave_1D

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.char_decomp=0
    else:
        solver = pyclaw.ClawSolver1D(rs)

    solver.bc_lower[0] = pyclaw.BC.wall
    solver.bc_upper[0] = pyclaw.BC.extrap
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    xlower = 0.
    xupper = L


    mx=int(round(xupper-xlower))*cells_per_layer
    x = pyclaw.Dimension(xlower,xupper,mx,name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain,solver.num_eqn,num_aux=1)

    #Set global parameters
    state.problem_data = {}
    state.problem_data['t1']    = 10.0
    state.problem_data['gamma'] = 1.4
    state.problem_data['pstar'] = 1.0
    state.problem_data['vstar'] = 1.0

    state.problem_data['tw1']   = 10.0
    state.problem_data['a1']    = 0.1
    state.problem_data['alpha'] = alpha
    state.problem_data['SA'] = SA
    state.problem_data['SB'] = SB
    state.problem_data['trtime'] = 999999999.0
    state.problem_data['trdone'] = False

    #Initialize q and aux
    xc=state.grid.x.centers
    state.aux[:,:] = setaux(xc,SB,SA,alpha,xlower=xlower,xupper=xupper)
    qinit(state,amp=amp,xupper=xupper)


    solver.max_steps = 5000000
    solver.fwave = True 
    #solver.before_step = b4step 

    claw = pyclaw.Controller()
    claw.output_style = 1
    claw.num_output_times = nout
    claw.tfinal = tfinal
    claw.write_aux_init = True
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.outdir = outdir
    claw.keep_copy = True

    return claw


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup)
