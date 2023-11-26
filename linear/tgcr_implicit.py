import numpy as np

from domain_specific.evalf import evalf


def tgcr_matrix_free(fhand, xf, pf, uf, b, tolrGCR, MaxItersGCR, epsMF, verbose=True):
    """
    Generalized conjugate residual method for solving [df/dx] x = b 
     using a matrix-free (i.e. matrix-implicit) technique
     INPUTS
     eval_f     : name of the function that evaluates f(xf,pf,uf)
     xf         : state vector where to evaluate the Jacobian [df/dx]
     pf         : structure containing parameters used by eval_f
     uf         : input needed by eval_f
     b          : right hand side of the linear system to be solved
     tolrGCR    : convergence tolerance, terminate on norm(b - Ax) / norm(b) < tolrGCR
     MaxItersGCR: maximum number of iterations before giving up
     epsMF      : finite difference perturbation for Matrix Free directional derivative
     OUTPUTS
     x          : computed solution, returns null if no convergence
     r_norms    : vector containing ||r_k||/||r_0|| for each iteration k
    
     EXAMPLE:
     [x, r_norms] = tgcr_MatrixFree(eval_f,x0,b,tolrGCR,MaxItersGCR,epsMF)
    """

    # Generate the initial guess for x (zero)
    x = np.zeros_like(b)

    # Set the initial residual to b - Ax^0 = b
    r = b.copy()
    r_norms = [np.linalg.norm(r, 2)]

    p_full = []
    Mp_full = [] 
    k = 0
    while (r_norms[k]/r_norms[0] > tolrGCR) and (k <= MaxItersGCR):
        # Use the residual as the first guess for the new search direction and multiply by M
        p = r.copy()
        epsilon=2*epsMF*np.sqrt(1+np.linalg.norm(xf,np.inf))/np.linalg.norm(p,np.inf) #NITSOL normal. great

        fepsMF  = fhand(xf+epsilon*p, t=None, p=pf, u=uf)
        f0 = fhand(xf, t=None, p=pf, u=uf)
        Mp = (fepsMF - f0)/epsilon
        assert np.linalg.norm(Mp) > 0, "tgcr_matrix_free: f0 and fepsMF are the same"
        # Make the new Ap vector orthogonal to the previous Mp vectors,
        # and the p vectors M^TM orthogonal to the previous p vectors.        
        if k>0:
            for j in range(k):
                beta = np.dot(Mp, Mp_full[j])
                p -= beta * p_full[j]
                Mp -= beta * Mp_full[j]

        # Make the orthogonal Mp vector of unit length, and scale the
        # p vector so that M * p  is of unit length
        norm_Mp = np.linalg.norm(Mp, 2)
        if norm_Mp == 0:
            print(f'GCR breakdown at step{k}!!!\n')
            print(r_norms)
            assert False
            break
        Mp = Mp/norm_Mp
        p = p/norm_Mp

        p_full.append(p)
        Mp_full.append(Mp)

        # Determine the optimal amount to change x in the p direction by projecting r onto Ap
        alpha = np.dot(r, Mp)

        # Update x and r
        x = x + alpha * p
        r = r - alpha * Mp

        # Save the norm of r
        r_norms.append(np.linalg.norm(r, 2))

        # Check convergence
        if r_norms[-1] < tolrGCR * r_norms[0]:
            break
        k += 1

    if r_norms[-1] > tolrGCR * r_norms[0]:
        print('GCR NONCONVERGENCE!!!\n')
        x = None
    elif verbose:
        print(f'GCR converged in {k+1} iterations')

    #print("GCR returned", x, fhand(x, None, pf, uf))
    r_norms = np.array(r_norms) / r_norms[0]
    return x, r_norms


def tgcr_find_root(x0, p, u, tolrGCR=1e-4, MaxItersGCR=100_000, eps=1e-5, eval_f=evalf, verbose=True):
    b = -eval_f(x0, t=None, p=p, u=u)
    max_iters = np.minimum(MaxItersGCR, np.prod(x0.shape))
    return tgcr_matrix_free(
        fhand=eval_f,
        xf=x0,
        pf=p,
        uf=u,
        b=b,
        tolrGCR=tolrGCR,
        MaxItersGCR=max_iters,
        epsMF=eps,
        verbose=verbose)