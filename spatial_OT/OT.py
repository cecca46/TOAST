import numpy as np
import warnings


def compute_transport(G0, epsilon, alpha, C1, C2, p, q, M, C3=None, C4=None, tol=1e-9, max_iter=1000):
    """
    Compute optimal transport using the Sinkhorn-Knopp algorithm.

    Parameters:
        G0 (ndarray): Initial transport matrix.
        epsilon (float): Regularization parameter.
        alpha (float): Weight parameter for cost function.
        C1, C2 (ndarray): Cost matrices for source and target distributions.
        p, q (ndarray): Marginal distributions for source and target.
        M (ndarray): Additional cost matrix for primary term.
        C3 (ndarray, optional): Additional cost matrix for coherence term (if applicable).
        coherence (bool, optional): Whether to include the coherence term.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        T (ndarray): Final transport matrix.
    """

    # Transformation functions for cost matrices
    def f1(a):
        return a**2

    def f2(b):
        return b**2

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    # Initialize transport matrix and variables
    T = G0
    cpt = 0  # Counter for iterations
    err = 1  # Error initialization

    # Precompute constants for the cost update
    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    constC1 = np.dot(np.dot(fC1, np.reshape(p, (-1, 1))), np.ones((1, len(q))))
    constC2 = np.dot(np.ones((len(p), 1)), np.dot(np.reshape(q, (1, -1)), fC2.T))
    constC = constC1 + constC2

    # Iterative process to compute optimal transport
    while err > tol and cpt < max_iter:
        Tprev = T.copy()  # Store previous transport matrix for error calculation

        # Compute the tensor term for cost update
        A = -np.dot(np.dot(hC1, T), hC2.T)
        tens = 2 * (constC + A)

        # Add coherence term if enabled
        if C3 is None and C4 is None:
            tens = alpha * tens + (1 - alpha) * M
        elif C3 is not None and C4 is None:
            tens = alpha * 0.5 * (tens + C3) + (1 - alpha) * M
        elif C3 is None and C4 is not None:
            tens = alpha * 0.5 * (tens + C4) + (1 - alpha) * M
        else:
            tens = alpha * (1 / 3.0) * (tens + C3 + C4) + (1 - alpha) * M

        # Update the transport matrix using Sinkhorn-Knopp
        T = sinkhorn_knopp(p, q, tens, epsilon)

        # Compute error every 10 iterations
        if cpt % 10 == 0:
            err = np.linalg.norm(T - Tprev)

        cpt += 1  # Increment iteration counter

    return T


def sinkhorn_knopp(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False, warn=True, warmstart=None):
    """
    Sinkhorn-Knopp algorithm for entropy-regularized optimal transport.

    Parameters:
        a, b (ndarray): Marginal distributions for source and target.
        M (ndarray): Cost matrix.
        reg (float): Regularization parameter.
        numItermax (int): Maximum number of iterations.
        stopThr (float): Convergence threshold.
        verbose (bool): Whether to display progress.
        log (bool): Whether to log errors.
        warn (bool): Whether to show warnings for convergence issues.
        warmstart (tuple, optional): Initial values for u and v (if provided).

    Returns:
        ndarray: Final transport matrix.
    """

    # Initialize marginal distributions if not provided
    if len(a) == 0:
        a = np.full((M.shape[0],), 1.0 / M.shape[0])
    if len(b) == 0:
        b = np.full((M.shape[1],), 1.0 / M.shape[1])

    # Initialization of u, v, and kernel matrix K
    if warmstart is None:
        u = np.ones(len(a)) / len(a)
        v = np.ones(len(b)) / len(b)
    else:
        u, v = np.exp(warmstart[0]), np.exp(warmstart[1])

    K = np.exp(-M / reg)  # Kernel matrix
    Kp = (1 / a).reshape(-1, 1) * K

    # Iterative scaling
    for ii in range(numItermax):
        uprev, vprev = u.copy(), v.copy()  # Save previous values for error calculation

        # Update scaling vectors u and v
        KtransposeU = np.dot(K.T, u)
        v = b / KtransposeU
        u = 1.0 / np.dot(Kp, v)

        # Check for numerical errors
        if (
            np.any(KtransposeU == 0)
            or np.any(np.isnan(u))
            or np.any(np.isnan(v))
            or np.any(np.isinf(u))
            or np.any(np.isinf(v))
        ):
            warnings.warn(f"Warning: numerical errors at iteration {ii}")
            u, v = uprev, vprev  # Restore previous values
            break

        # Compute error every 10 iterations
        if ii % 10 == 0:
            tmp2 = np.einsum("i,ij,j->j", u, K, v)  # Marginal comparison
            err = np.linalg.norm(tmp2 - b)  # Violation of marginal
            if err < stopThr:
                break  # Convergence achieved

    # Final transport matrix
    T = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    return T