import numpy as np

def analytical_solution(nx, ny):
    '''Analytical solution for laplace equation with given boundary conditions'''
    y = np.linspace(0, 1, ny)
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
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx): 
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, jr] + u[i, jl])
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u, errors


def gauss_seidel(nx, ny, max_iter = 10000, tol = 1e-5):
    '''Gauss-Seidel method for laplace equation'''
    u = np.zeros((nx, ny))

    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl])
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary
        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol :
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u, errors

def successive_over_relaxation(nx, ny, w, max_iter = 10000, tol = 1e-5):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * w * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl]) + (1 -w) * u[i, j]
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u, errors

def successive_over_relaxation_it(nx, ny, w, max_iter = 10000, tol = 1e-5):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary
                u_new[i, j] = 0.25 * w * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl]) + (1 -w) * u[i, j]
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            return it
            
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return max_iter



def successive_over_relaxation_sink(nx, ny, w, max_iter = 10000, tol = 1e-5,sinks=[[25,75,25,75,True]]):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    # Set sink conditions
    sink_exist = sinks is not None
    def is_sink(i,j):
        if not sink_exist:
            return False
        for sink in sinks:
            x1,x2,y1,y2,positive = sink
            if x1 <= j < x2 and y1 <= i < y2:
                if positive:
                    u_new[i,j] = 1
                else:                   
                    u_new[i,j] = 0
                return True
        return False
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                if not is_sink(i,j):    
                    jr = (j + 1) % (nx - 1)  # right boundary
                    jl = (j - 1) % (nx - 1)  # left boundary
                    u_new[i, j] = 0.25 * w * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl]) + (1 -w) * u[i, j]
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u, errors


def successive_over_relaxation_sink_it(nx, ny, w, max_iter = 10000, tol = 1e-5,sinks=[[25,75,25,75,True]]):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    # Set sink conditions
    sink_exist = sinks is not None
    def is_sink(i,j):
        if not sink_exist:
            return False
        for sink in sinks:
            x1,x2,y1,y2,positive = sink
            if x1 <= j < x2 and y1 <= i < y2:
                if positive:
                    u_new[i,j] = 1
                else:                   
                    u_new[i,j] = 0
                return True
        return False
    
    errors = []
    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                if not is_sink(i,j):    
                    jr = (j + 1) % (nx - 1)  # right boundary
                    jl = (j - 1) % (nx - 1)  # left boundary
                    u_new[i, j] = 0.25 * w * (u[i+1, j] + u_new[i-1, j] + u[i, jr] + u_new[i, jl]) + (1 -w) * u[i, j]
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            return it
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return max_iter



def successive_over_relaxation_insulating(nx, ny, w, max_iter = 10000, tol = 1e-5,sinks=[[25,75,25,75]]):
    '''Successive Over-Relaxation method for laplace equation'''
    u = np.zeros((nx, ny))
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 0  # Top boundary
    u[-1, :] = 1  # Bottom boundary
    u_new[0, :] = 0  # Top boundary
    u_new[-1, :] = 1  # Bottom boundary
    
    # Set sink conditions
    sink_exist = sinks is not None
    def is_sink(i,j):
        if not sink_exist:
            return False
        for sink in sinks:
            x1,x2,y1,y2 = sink
            if x1 <= j < x2 and y1 <= i < y2:
                return True
        return False
    
    errors = []

    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, ny - 1):
            for j in range(nx):
                if is_sink(i,j):    
                     continue  
                 
                jr = (j + 1) % (nx - 1)  # right boundary
                jl = (j - 1) % (nx - 1)  # left boundary

                def get_value(i1,j1):
                    if is_sink(i1,j1):
                        return u_new[i,j]
                    else:
                        return u_new[i1,j1]
                a = get_value(i+1,j)
                b = get_value(i-1,j)
                c = get_value(i,jr)
                d = get_value(i,jl)
                u_new[i, j] = 0.25 * w * (a + b + c + d) + (1 -w) * u[i, j]
        u_new[0, :] = 0  # Top boundary
        u_new[-1, :] = 1  # Bottom boundary

        errors.append(np.max(np.abs(u_new - u)))
        # Check for convergence
        if np.max(np.abs(u_new - u)) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u, errors