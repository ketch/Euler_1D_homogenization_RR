import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
ifft = np.fft.ifft
fft = np.fft.fft
from ipywidgets import IntProgress
from IPython.display import display
import time

pstar = 1.
vstar = 1.
gamma = 1.4

def bracket(f):
    """
    Returns a function that computes the bracket of a given function f.

    Parameters:
    f (function): The function to compute the bracket of.

    Returns:
    function: A function that computes the bracket of f.
    """
    mean = quad(f,0,1)[0]
    brace = lambda y: f(y)-mean
    brack_nzm = lambda y: quad(brace,0,y)[0]
    mean_bracket = quad(brack_nzm,0,1)[0]
    def brack(y):
        return quad(brace,0,y)[0] - mean_bracket
    return brack

def spectral_representation(x0,uhat,xi):
    """
    Returns a truncated Fourier series representation of a function.

    Parameters:
    x0 (float): The left endpoint of the domain of the function.
    uhat (numpy.ndarray): The Fourier coefficients of the function.
    xi (numpy.ndarray): The vector of wavenumbers.

    Returns:
    u_fun: A vectorized function that represents the Fourier series.
    """
    u_fun = lambda y : np.real(np.sum(uhat*np.exp(1j*xi*(y+x0))))/len(uhat)
    u_fun = np.vectorize(u_fun)
    return u_fun

def fine_resolution(f, n, x, xi):
    """
    Interpolates a periodic function `f` onto a finer grid of `n` points using a Fourier series.

    Parameters:
    -----------
    f : function
        The function to be interpolated.
    n : int
        The number of points in the finer grid.
    x : array-like
        The original grid of `f`.
    xi : array-like
        The Fourier modes.

    Returns:
    --------
    x_fine : array-like
        The finer grid of `n` points.
    f_spectral : function
        The Fourier interpolation `f` on the finer grid.
    """
    fhat = fft(f)
    f_spectral = spectral_representation(x[0],fhat,xi)
    x_fine = np.linspace(x[0],x[-1],n)
    return x_fine, f_spectral(x_fine)

def rk3(u,xi,rhs,dt,du,params=None):
    """
    Third-order Runge-Kutta time-stepping method for solving ODEs.

    Parameters:
    u (numpy.ndarray): The current solution.
    xi (numpy.ndarray): The spatial grid.
    rhs (function): The right-hand side of the ODE system.
    dt (float): The time step size.
    du (float): The spatial step size.
    params (dict, optional): Additional parameters to pass to the RHS function.

    Returns:
    numpy.ndarray: The updated solution at the next time step.
    """
    y2 = u + dt*rhs(u,du,xi,**params)
    y3 = 0.75*u + 0.25*(y2 + dt*rhs(y2,du,xi,**params))
    u_new = 1./3 * u + 2./3 * (y3 + dt*rhs(y3,du,xi,**params))
    return u_new

def rkm(u,xi,rhs,dt,du,fy,method,params=None):
    A = method.A
    b = method.b
    for i in range(len(b)):
        y = u.copy()
        for j in range(i):
            y += dt*A[i,j]*fy[j,:,:]
        fy[i,:,:] = rhs(y,du,xi,**params)
    #u_new = u + dt*sum([b[i]*fy[i,:,:] for i in range(len(b))])
    u_new = u + dt*np.sum(b[:,np.newaxis,np.newaxis]*fy, axis=0) # faster
    return u_new


def xxt_rhs(q, dq, xi, **params):
    """
    Solves the BBM-like equation for a given set of parameters.

    Args:
        u (ndarray): Array of shape (2, N) containing the values of eta and q.
        du (ndarray): Array of shape (2, N) containing the derivatives of eta and q.
        xi (ndarray): Array of shape (N,) containing the values of xi.
        **params: Dictionary containing the values of the parameters H1, H2, H3, H4, alpha1, alpha2, alpha3, alpha4,
                  alpha5, alpha6, alpha7, alpha8, alpha9, alpha10, alpha11, C3, C11, delta, and order4.

    Returns:
        ndarray: Array of shape (2, N) containing the derivatives of eta and q.
    """
    delta = params['delta']
    mu = params['mu']
    zeta = params['zeta']
    K1 = params['K1']
    kappa = params['kappa']
    order4 = params['order4']

    p = q[0,:]
    u   = q[1,:]

    G   = kappa*p**((gamma+1)/gamma)
    G1  = kappa*(gamma+1)/gamma * p**(1/gamma)
    G2 = kappa*(gamma+1)/gamma**2 * p**(1/gamma-1)
    G3 =  kappa*(1-gamma**2)/gamma**3 * p**(1/gamma-2)
    G4 =  kappa*(1-2*gamma)*(1-gamma**2)/gamma**4 * p**(1/gamma-3)
    #p = pstar*np.exp(-S)*(Vstar/V)**gamma
    phat = fft(p)
    uhat = fft(u)
    
    p_x = np.real(ifft(1j*xi*phat))
    u_x = np.real(ifft(1j*xi*uhat))
    p_xx = np.real(ifft(-xi**2*phat))
    u_xx = np.real(ifft(-xi**2*uhat))
    p_xxx = np.real(ifft(-1j*xi**3*phat))
    u_xxx = np.real(ifft(-1j*xi**3*uhat))
    p_xxxx = np.real(ifft(xi**4*phat))
    u_xxxx = np.real(ifft(xi**4*uhat))

    beta = G*u_xxx + G1*p_xx*u_x + 2*G1*p_x*u_xx + G2*p_x**2*u_x

   # 4th-order nonlinear terms
    NN = (zeta/K1**3-mu**2)/K1 * ( beta*(2*G1/G*p_xx - G2/G*p_x**2) - 6*G1*p_xxx*u_xx - G1*p_xx*u_xxx - 6*G1**2/G*p_x**2*u_xxx \
                                -(8*G1**2/G+6*G2)*p_x*p_xx*u_xx - 6*G2*p_x*p_xxx*u_x - 6*G1*G2/G*p_x**3*u_xx \
                                + beta/K1*(G2/G-2*G1**2)*u_x**2 + (2*G1**2-6*G*G2)/K1*u_x*u_xx**2 \
                                +(2*G1**2-G*G2)/K1*u_x**2*u_xxx + (2*G1**3/G-G1*G2)/K1*p_xx*u_x**3 \
                                + (-9*G1*G2/G-3*G3)*p_x**2*p_xx*u_x - 2*G1*G3/G*p_x**4*u_x \
                                + (4*G1**3/G-4*G1*G2-6*G*G3)/K1*p_x*u_x**2*u_xx \
                                + (2*G1**2*G2/G - 2*G2**2 - 2*G1*G3 - G*G4)/K1*p_x**2*u_x**3 + G1*p_xxxx*u_x \
                                - 2*G1*p_x*u_xxxx - 2*G1**2/G*p_xx**2*u_x ) \
                                + zeta*G2/(2*K1**4)*p_xx**2*u_x
    NN = -NN # Move terms to RHS
   # NN = 0

    if order4:
        dp = (-G*u_x + delta**2*mu*G1*p_xx*u_x ) /K1 + delta**4*NN
        dphat = fft(dp)
        dp = np.real(ifft(dphat/(1 + xi**2*delta**2*mu + xi**4*delta**4*(zeta/K1**3 - mu**2))))
    else:
        dp = (-G*u_x + delta**2*mu*G1*p_xx*u_x ) /K1
        dphat = fft(dp)
        dp = np.real(ifft(dphat/(1 + xi**2*delta**2*mu)))
    du = -p_x

    dq[0,:] = dp
    dq[1,:] = du
    return dq


def homogenized_coefficients(SA,SB,delta,alpha):
    """
    Computes homogenized coefficients for a given periodic entropy profile.

    Parameters:

    Returns:
    dict: A dictionary containing the homogenized coefficients.
    """
    params = {}

    KiA = np.exp(SA/gamma)
    KiB = np.exp(SB/gamma)
    K1 = alpha*KiA + (1-alpha)*KiB
    #K1 = (KiA+KiB)/2
    mu = alpha**2*(1-alpha)**2*(KiA-KiB)**2/(12*(alpha*KiA+(1-alpha)*KiB)**2)
    #mu = (KiA-KiB)**2/192
    #mu = mu/K1**2
    zeta = alpha**2*(1-alpha)**2*(KiA-KiB)**2/720 * ((1-3*alpha+8*alpha**2-6*alpha**3)*KiB + (5*alpha+6*alpha**3-10*alpha**2)*KiA)
    #zeta = (KiA-KiB)**2*(KiA+KiB)/15360
    params['K1'] = K1
    params['mu'] = mu
    params['zeta'] = zeta
    params['delta'] = delta
    cstarsq = gamma*pstar/vstar
    params['kappa'] = cstarsq/pstar**((gamma+1)/gamma)

    return params


def solve_psystem_homog(p_amp=0.3, SA=1, SB=4, width=3.0, L=200,tmax=100.,m=256, dtfac=0.5,
                make_anim=True,num_plots=100,order4=True, alpha=1./8):
    """
    Solve the homogenized p-system using Fourier spectral collocation in space
    and a RK method in time, on the domain (-L/2,L/2).

    Parameters:
    -----------
    p_amp : float, optional
        Amplitude of the initial pressure wave. Default is 0.3.
    width : float, optional
        Width of the initial wave. Default is 3.0.
    L : float, optional
        Length of the domain. Default is 200.
    tmax : float, optional
        Maximum time to run the simulation. Default is 100.
    m : int, optional
        Number of grid points. Default is 256.
    dtfac : float, optional
        Time step factor. Default is 0.5.
    make_anim : bool, optional
        Whether to create an animation of the simulation. Default is True.

    Returns:
    --------
    x : numpy.ndarray
        Array of grid points.
    xi : numpy.ndarray
        Array of wavenumbers.
    p : list
        List of arrays of pressure at each time step.
    u : list
        List of arrays of velocity at each time step.
    anim : matplotlib.animation.FuncAnimation or None
        Animation of the simulation, if make_anim is True. Otherwise, None.
    """
    # ================================
    # Compute homogenized coefficients
    # ================================

    delta = 1.0
    params = homogenized_coefficients(SA,SB,delta,alpha)
    params['order4'] = order4
     # ================================

    # Grid
    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    from nodepy import rk
    method = rk.loadRKM('BS5').__num__()
    c = np.sqrt(gamma*pstar/vstar)
    dt = dtfac * 1.73/np.max(xi) / c
    fy = np.zeros((len(method),2,m))

    
    p0 = pstar + p_amp * np.exp(-x**2 / width**2)
    u0 = np.zeros_like(p0)
    q = np.zeros((2,len(x)))

    q[0,:] = p0
    q[1,:] = u0

    dq = np.zeros_like(q)

    plot_interval = tmax/num_plots
    steps_between_plots = int(round(plot_interval/dt))
    dt = plot_interval/steps_between_plots
    nmax = num_plots*steps_between_plots

    fig = plt.figure(figsize=(12,8))
    axes = fig.add_subplot(111)
    line, = axes.plot(x,q[0,:],lw=2)
    axes.set_xlabel(r'$x$',fontsize=30)
    plt.close()

    p = [q[0,:].copy()]
    u = [q[1,:].copy()]
    tt = [0]
    
    f = IntProgress(min=0, max=num_plots) # instantiate the bar
    display(f) # display the bar


    for n in range(1,nmax+1):
        q_new = rkm(q,xi,xxt_rhs,dt,dq,fy,method,params)
            
        q = q_new.copy()
        t = n*dt

        # Plotting
        if np.mod(n,steps_between_plots) == 0:
            f.value += 1
            p.append(q[0,:].copy())
            u.append(q[1,:].copy())
            tt.append(t)
        
    def plot_frame(i):
        phat = np.fft.fft(p[i])
        p_spectral = spectral_representation(x[0],phat,xi)
        x_fine = np.linspace(x[0],x[-1],5000)
        line.set_data(x_fine,p_spectral(x_fine))
        axes.set_title('t= %.2e' % tt[i])

    if make_anim:
        anim = matplotlib.animation.FuncAnimation(fig, plot_frame,
                                           frames=len(p), interval=100)
        anim = HTML(anim.to_jshtml())
    else:
        anim = None

    return x, xi, p, u, anim
