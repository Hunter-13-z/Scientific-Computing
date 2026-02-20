import numpy as np

def analytical_solution(nx, ny):
    '''Analytical solution for laplace equation with given boundary conditions'''
    y = np.linspace(1, 0, ny)
    u = np.tile(y[:,None], (1,nx))  
    return u


def compare_to_analytic(u):
    ny, nx = u.shape
    u_exact = analytical_solution(nx, ny)

    err = u[1:-1,:] - u_exact[1:-1,:]
    linf = np.max(np.abs(err))                          # max error
    rmse = np.sqrt(np.mean(err**2))                     # root mean square error

    return linf, rmse


def jacobi(nx, ny, max_iter = 10000, tol = 1e-5):
    '''Jacobi method for laplace equation'''
    u = np.zeros((ny, nx))
    
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 1  # Top boundary
    u[-1, :] = 0  # Bottom boundary
    u_new[0, :] = 1  # Top boundary
    u_new[-1, :] = 0  # Bottom boundary
    

    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx): 
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, jr] + u[i, jl])
        u_new[0, :] = 1  # Top boundary
        u_new[-1, :] = 0  # Bottom boundary

        # Check for convergence
        if np.linalg.norm(u_new - u) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u


def gauss_seidel(nx, ny, max_iter = 10000, tol = 1e-5):
    '''Gauss-Seidel method for laplace equation'''
    u = np.zeros((nx, ny))

    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 1  # Top boundary
    u[-1, :] = 0  # Bottom boundary
    u_new[0, :] = 1  # Top boundary
    u_new[-1, :] = 0  # Bottom boundary
    

    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl])
        u_new[0, :] = 1  # Top boundary
        u_new[-1, :] = 0  # Bottom boundary
        # Check for convergence
        if np.linalg.norm(u_new - u) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u

def successive_over_relaxation(nx, ny, w, max_iter = 10000, tol = 1e-5, omega = 1.5):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 1  # Top boundary
    u[-1, :] = 0  # Bottom boundary
    u_new[0, :] = 1  # Top boundary
    u_new[-1, :] = 0  # Bottom boundary
    

    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * w * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl]) + (1 -w) * u[i, j]
        u_new[0, :] = 1  # Top boundary
        u_new[-1, :] = 0  # Bottom boundary
        # Check for convergence
        if np.linalg.norm(u_new - u) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u