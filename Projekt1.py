import numpy as np
from scipy.sparse import kron, csc_matrix, eye, vstack
from scipy.sparse.linalg import inv
from math import sqrt, ceil
import operators as ops
import matplotlib.pyplot as plt
from numba import jit
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def simulation_6(mx=101, time_step = 0.01, end_time = 1.8, order=2, boundry = None, show_animation = True, boundary_conditions = "Neuman", sim_7 = False):
    # Method parameters

    # Type of method.
    # 1 - Projection
    # 0 - SAT

    method = 1


    # Model parameters
    T = end_time # end time

    beta_l = 1
    beta_r = 1

    alpha_l = 1
    alpha_r = 1

    gamma_l = 1
    gamma_r = 1

    # Domain boundaries
    xl = -1
    xr = 1
    c = 1

    # Space discretization

    hx = (xr - xl)/(mx-1)
    xvec = np.linspace(xl,xr,mx)

    if order == 2:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(mx,hx)
    elif order == 4:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(mx,hx)
    elif order == 6:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(mx,hx)
    else:
        raise NotImplementedError('Order not implemented.')

    #Fix correct transpose
    e_l = e_l.T
    e_r = e_r.T
    d1_l = d1_l.T
    d1_r = d1_r.T

    I_m = np.eye(mx)
    if boundary_conditions == "Dirichlet":
        L = vstack((beta_l*e_l.T,beta_r*e_r.T), format="csc")
        P = I_m - HI@L.T@inv(L@HI@L.T)@L
        A = P@((c**2)*D2)@P
    elif boundary_conditions == "Neuman":
          M = e_r @ d1_r.T- H @ D2 - e_l @ d1_l.T
          A = -(c**2)*HI@M
    elif boundary_conditions == "ABC":
          M = e_r @ d1_r.T- H @ D2 - e_l @ d1_l.T
          A = -c**2 * HI @ M
          B = -c**2 * HI @ (e_r @ e_r.T + e_l @ e_l.T)

    def gauss(x):
        rw = 5
        return np.exp(-(rw*x)**2)

    v = np.hstack(gauss(xvec))
    w = np.hstack(np.zeros_like(v))

    def rhs(vw):
        v = vw[:mx]
        w = vw[mx:]
        if boundry == "ABC":
              dvdt = w
              dwdt = A @ v + B @ w
              dwdt = np.array(dwdt).flatten()
        else:
        
            dvdt = w
            dwdt = A @ v
            dwdt = np.array(dwdt).flatten()

            return np.hstack([dvdt, dwdt])

    # Concatenate v and w for RK4
    vw = np.hstack([v, w])
    # Time discretization
    CFL = 0.2
    #ht_try = CFL*hx**2
    ht_try = time_step
    mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
    tvec,ht = np.linspace(0,T,mt,retstep=True)


    #Plot
    if show_animation == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        [line1] = ax.plot(xvec, v, label='Solution with ' +  boundary_conditions + ' BC')
        plt.legend()
        ax.set_xlim([xl, xr])
        ax.set_ylim([-1, 1.5])
        title = plt.title("t = " + "{:.2f}".format(0))
        plt.draw()
        plt.pause(0.5)

    

    # Runge-Kutta 4
    t = 0
    solutions_at_times = {}
    for tidx in range(mt-1):
        vw = step(rhs, vw, t, ht)
        t = tvec[tidx + 1]
        if show_animation == True:
            #Uppdate the plot
            if tidx % ceil(5) == 0 or tidx == mt-2:
                    v0 = vw[0:mx]
                    line1.set_ydata(v0)
                    title.set_text("t = " + "{:.2f}".format(tvec[tidx+1]))
                    plt.draw()
                    plt.pause(1e-3)
        #Calculate errors and more
        elif show_animation == False:
            if np.isclose(t, 0.2) or np.isclose(t, 0.5) or np.isclose(t, 0.7) or np.isclose(t, 1.8):
                solutions_at_times[t] = np.copy(vw[:mx])
        
    if show_animation == True: plt.show()
    if sim_7 == True:
         return hx , vw[:mx], xvec 

def step(f, v, t, dt):
        """Take one RK4 step. Return updated solution and time.
        f: Right-hand-side function: dv/dt = f(v)
        v: current solution
        t: current time
        dt: time step 
        """

        # Compute rates k1-k4
        k1 = dt*f(v)
        k2 = dt*f(v + 0.5*k1)
        k3 = dt*f(v + 0.5*k2)
        k4 = dt*f(v + k3)

        # Update solution and time
        v = v + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        return v

def simulation_7():
    c = 1
    mx_list = [51, 101, 201, 301]
    order_list = [4 , 6]
    boundary_conditions_list = ["Dirichlet", "Neuman"]
    k = 0.1
    end_time_list = [1.8, 2.1]
    Convergens_rate_order_4 = pd.DataFrame(index=["T = 1.8",'Dirichlet BC 1', 'Neumann BC 1', "T = 2.1", 'Dirichlet BC 2','Neumann BC 2'], columns=['50/100', '100/200', '200/300'])
    Convergens_rate_order_6 = pd.DataFrame(index=["T = 1.8",'Dirichlet BC 1', 'Neumann BC 1', "T = 2.1", 'Dirichlet BC 2','Neumann BC 2'], columns=['50/100', '100/200', '200/300'])

    l2_norm_error_order_4 = pd.DataFrame(index=["T = 1.8",'Dirichlet BC 1', 'Neumann BC 1', "T = 2.1", 'Dirichlet BC 2','Neumann BC 2'], columns=['50', '100', '200', '300'])
    l2_norm_error_order_6 = pd.DataFrame(index=["T = 1.8",'Dirichlet BC 1', 'Neumann BC 1', "T = 2.1", 'Dirichlet BC 2','Neumann BC 2'], columns=['50', '100', '200', '300'])
    for i in range(0,len(mx_list)-1):
        m1 = mx_list[i]
        m2 = mx_list[i+1]
        time_step1 = k*(2/m1)
        time_step2 = k*(2/m2)

        #Order 4
        #Dirichlet with endtime 1.8 and order 4
        Convergens_rate_order_4.loc["Dirichlet BC 1", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[0], order_list[0], boundary_conditions_list[0])
        Convergens_rate_order_4.loc["Neumann BC 1", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[0], order_list[0], boundary_conditions_list[1])
        #Neuman with endtime 2.1 and order 4
        Convergens_rate_order_4.loc["Dirichlet BC 2", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[1], order_list[0], boundary_conditions_list[0])
        Convergens_rate_order_4.loc["Neumann BC 2", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[1], order_list[0], boundary_conditions_list[1])

        #Order 6
        #Dirichlet with endtime 1.8 and order 4
        Convergens_rate_order_6.loc["Dirichlet BC 1", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[0], order_list[1], boundary_conditions_list[0])
        Convergens_rate_order_6.loc["Neumann BC 1", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[0], order_list[1], boundary_conditions_list[1])
        #Neuman with endtime 2.1 and order 6
        Convergens_rate_order_6.loc["Dirichlet BC 2", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[1], order_list[1], boundary_conditions_list[0])
        Convergens_rate_order_6.loc["Neumann BC 2", str(m1-1)+"/" + str(m2-1)] = convergens(m1, m2, time_step1, time_step2, end_time_list[1], order_list[1], boundary_conditions_list[1])

    for i in mx_list:
        # l2 error calculations

        # t = 1.8
        #Dirichlet order 4
        l2_norm_error_order_4.loc["Dirichlet BC 1", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[0], order_list[0], boundary_conditions_list[0])
        #Neumann order 4
        l2_norm_error_order_4.loc["Neumann BC 1", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[0], order_list[0], boundary_conditions_list[1])
        
        #t = 2.1
        #Dirichlet order 4
        l2_norm_error_order_4.loc["Dirichlet BC 2", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[1], order_list[0], boundary_conditions_list[0])
        #Neumann order 4
        l2_norm_error_order_4.loc["Neumann BC 2", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[1], order_list[0], boundary_conditions_list[1])

         # t = 1.8
        #Dirichlet order 6
        l2_norm_error_order_6.loc["Dirichlet BC 1", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[0], order_list[1], boundary_conditions_list[0])
        #Neumann order 6
        l2_norm_error_order_6.loc["Neumann BC 1", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[0], order_list[1], boundary_conditions_list[1])
        
        #t = 2.1
        #Dirichlet order 6
        l2_norm_error_order_6.loc["Dirichlet BC 2", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[1], order_list[1], boundary_conditions_list[0])
        #Neumann order 6
        l2_norm_error_order_6.loc["Neumann BC 2", str(i-1)] = l2_norm_error(i,time_step1, end_time_list[1], order_list[1], boundary_conditions_list[1])

    filename = "Convergens study.xlsx"
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        Convergens_rate_order_4.to_excel(writer, sheet_name='Convergens for order 4', index=True)
        Convergens_rate_order_6.to_excel(writer, sheet_name='Convergens for order 6', index=True)
        l2_norm_error_order_4.to_excel(writer, sheet_name='l2 norm error for order 4', index=True)
        l2_norm_error_order_6.to_excel(writer, sheet_name='l2 norm error for order 6', index=True)
    print("Calculations completed and an excel file has been created")

def exact_solution(x, t, c, r, boundry = None):
    def profile_1(x, t):
        return np.exp(-1*(((x - c*t)/r)**2))
    def profile_2(x, t):
        return -np.exp(-1*(((x + c*t)/r)**2))
    
    if boundry == "Neuman":
        u_exact = 0.5 * (profile_1(x + 2, t) - profile_2(x - 2, t))
    if boundry == "Dirichlet":
        u_exact = 0.5 * (-profile_1(x + 2, t) + profile_2(x - 2, t))
    return u_exact

#FÖRKLARA?

def l2_norm(vec, h):
    return np.sqrt(h)*np.sqrt(np.sum(vec**2))

def compute_error(u, u_exact, hx):
    #Compute discrete l2 error
    error_vec = u - u_exact
    relative_l2_error = l2_norm(error_vec, hx)/l2_norm(u_exact, hx)
    return relative_l2_error

def convergens(m1, m2, time_step1, time_step2, end_time, order, BC):
     print(f'Simulation running with: {[m1, m2, time_step1, time_step2, end_time, order, BC]}')
     hx1, v1, xvec1 = simulation_6(mx=m1, time_step = time_step1, end_time = end_time, order = order, show_animation = False, boundary_conditions = BC, sim_7 = True)
     hx2, v2, xvec2 = simulation_6(mx=m2, time_step = time_step2, end_time = end_time, order = order, show_animation = False, boundary_conditions = BC, sim_7 = True )
     u_exact1 = np.hstack(exact_solution(xvec1, end_time, 1, 0.2, BC))
     u_exact2 = np.hstack(exact_solution(xvec2, end_time, 1, 0.2, BC))
     err1 = compute_error(v1, u_exact1, hx1)
     err2 = compute_error(v2, u_exact2, hx2)
     q = np.log10((err1/err2))/np.log10((hx2/hx1))
     return -q

def l2_norm_error(m,time_step, end_time, order, BC):
    hx, v, xvec = simulation_6(mx=m, time_step = time_step, end_time = end_time, order = order, show_animation = False, boundary_conditions = BC, sim_7 = True )
    u_exact = np.hstack(exact_solution(xvec, end_time, 1, 0.2, BC))
    return compute_error(v, u_exact, hx)


def simulation_8(boundary_conditions, time_step, T):
    # Domain boundaries
    m = 201
    xl = -1
    xr = 1

    yl = -1
    yr = 1

    c = 1

    beta_l = 1
    beta_r = 1

    alpha_l = 1
    alpha_r = 1

    gamma_l = 1
    gamma_r = 1



    h_xy = (xr - xl)/(m-1)
    xvec = np.linspace(xl,xr,m)
    yvec = np.linspace(yl,yr,m)

    H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(m,h_xy)

    e_l = e_l.T
    e_r = e_r.T
    d1_l = d1_l.T
    d1_r = d1_r.T

    I_m = np.eye(m)
    if boundary_conditions == "Dirichlet":
        L = vstack((beta_l*e_l.T,beta_r*e_r.T), format="csc")
        P = I_m - HI@L.T@inv(L@HI@L.T)@L
        A = P@((c**2)*D2)@P
        A_2D = (kron(A, P) + kron(P, A))
    elif boundary_conditions == "Neuman":
        M = e_r @ d1_r.T- H @ D2 - e_l @ d1_l.T
        A = -(c**2)*HI@M
        A_2D = kron(A, I_m) + kron(I_m, A)

    def f(x, y):
        return np.exp(-100*(x**2 + y**2))

    v = np.array([f(x_val, y_val) for x_val in xvec for y_val in yvec])
    w = np.hstack(np.zeros_like(v))
    vw = np.hstack([v,w])

    def rhs2(vw):
        v = vw[:m**2]
        w = vw[m**2:]

        dvdt = w
        dwdt = A_2D @ v
        dwdt = np.array(dwdt).flatten()

        return np.hstack([dvdt, dwdt])
    
    CFL = 0.2

    #ht_try = CFL*hx**2
    #ht_try = time_step
    print(h_xy)
    ht_try =  0.001
    mt = int(ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
    tvec,ht = np.linspace(0,T,mt,retstep=True)


    X, Y = np.meshgrid(xvec, yvec)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection='3d')
   
    v0 = v.reshape((m, m))

    surface = ax.plot_surface(X, Y, v0, cmap='viridis')

    # Adding labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('V')
    ax.set_title('3D Color-coded Surface Plot of Vector v')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Adding color bar
    #cbar = fig.colorbar(surface, ax=ax, label='Value of v')
    plt.draw()
    plt.pause(1)
    print("hej3")

    desired_times = [i*0.01 for i in range(1,1000)]

    

    t = 0
    solutions_at_times = {}
    for tidx in range(mt-1):
        print(tidx)
        vw = step(rhs2, vw, t, ht)
        t = tvec[tidx + 1]
        #Uppdate the plot
        if np.isclose(t, desired_times).any():
       # if tidx % ceil(5) == 0 or tidx == mt-2:
                #v0 = vw[0:m**2]
                
                v0 = vw[:m*m].reshape((m, m))
                #v0 = array_to_matrix(v0, m)
                ax.clear()  # Clear the current axes
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('V')
                ax.set_title("t = {:.2f}".format(tvec[tidx + 1]))
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
                surface = ax.plot_surface(X, Y, v0, cmap='viridis')
                #fig.colorbar(surface, ax=ax, label='Value of v')
                plt.draw()
                print("New uppdate in plot")
                plt.pause(1e-3)
    print(mt)
    plt.show()


def main():
    #simulation_6(mx=101, time_step = 0.001, end_time = 3, order=4, show_animation = True, boundary_conditions = "ABC")
    #simulation_7()
    simulation_8("Neuman", 0.1, 20)
    #simulation_8("Dirichlet", 0.1, 20)

if __name__ == '__main__':
    main()
