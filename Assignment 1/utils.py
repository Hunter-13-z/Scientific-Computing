import numpy as np


def jacobi(nx, ny, max_iter = 1000, tol = 1e-5):
    '''Jacobi method for laplace equation'''
    # Initialize the solution array
    u = np.zeros((nx, ny))
    
    # Create a copy of the solution array to store updates
    u_new = np.zeros_like(u)
    
    # Set boundary conditions 
    u[0, :] = 1  # Top boundary
    u[-1, :] = 0  # Bottom boundary
    u_new[0, :] = 1  # Top boundary
    u_new[-1, :] = 0  # Bottom boundary
    

    # Iteratively update the solution
    for it in range(max_iter):
        # Update each interior point
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                jr = (j + 1) % (ny - 1)  # right boundary
                jl = (j - 1) % (ny - 1)  # left boundary
                u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, jr] + u[i, jl])
        
        # Check for convergence
        if np.linalg.norm(u_new - u) < tol:
            print(f'Converged after {it} iterations.')
            break
        
        # Update the solution for the next iteration
        u[:] = u_new[:]
    
    return u