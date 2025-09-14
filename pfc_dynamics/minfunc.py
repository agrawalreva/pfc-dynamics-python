"""
minFunc - unconstrained optimizer using line search strategy
Python port of minFunc optimization library
"""

import numpy as np
from scipy.optimize import minimize, line_search
from scipy.sparse import issparse

def minfunc(fun_obj, x0, options=None, *args):
    """
    Unconstrained optimizer using line search strategy
    
    Args:
        fun_obj: objective function
        x0: initial point
        options: optimization options
        *args: additional arguments
    
    Returns:
        x: optimal point
        f: function value
        exitflag: exit flag
        output: output information
    """
    if options is None:
        options = {}
    
    # set default options
    method = options.get('Method', 'lbfgs')
    max_iter = options.get('MaxIter', 1000)
    tol_fun = options.get('TolFun', 1e-6)
    display = options.get('Display', 'off')
    
    x = x0.copy()
    f_old = np.inf
    exitflag = 1
    
    # used scipy's minimize
    if method == 'lbfgs':
        result = minimize(fun_obj, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': tol_fun, 'disp': display == 'iter'})
    elif method == 'cg':
        result = minimize(fun_obj, x0, method='CG',
                         options={'maxiter': max_iter, 'gtol': tol_fun, 'disp': display == 'iter'})
    elif method == 'bfgs':
        result = minimize(fun_obj, x0, method='BFGS',
                         options={'maxiter': max_iter, 'gtol': tol_fun, 'disp': display == 'iter'})
    else:
        # fallback to BFGS
        result = minimize(fun_obj, x0, method='BFGS',
                         options={'maxiter': max_iter, 'gtol': tol_fun, 'disp': display == 'iter'})
    
    x = result.x
    f = result.fun
    exitflag = result.success
    
    output = {
        'iterations': result.nit,
        'funcCount': result.nfev,
        'algorithm': method,
        'message': result.message
    }
    
    return x, f, exitflag, output

def armijo_backtrack(fun_obj, x, f, g, d, options=None):
    """
    Armijo backtracking line search
    
    Args:
        fun_obj: objective function
        x: current point
        f: current function value
        g: current gradient
        d: search direction
        options: line search options
    
    Returns:
        alpha: step size
        f_new: new function value
        g_new: new gradient
    """
    if options is None:
        options = {}
    
    alpha = options.get('alpha', 1.0)
    c1 = options.get('c1', 1e-4)
    rho = options.get('rho', 0.5)
    max_iter = options.get('max_iter', 20)
    
    for i in range(max_iter):
        x_new = x + alpha * d
        f_new = fun_obj(x_new)
        
        # Armijo condition
        if f_new <= f + c1 * alpha * np.dot(g, d):
            return alpha, f_new, None
        
        alpha *= rho
    
    return alpha, f_new, None

def wolfe_line_search(fun_obj, x, f, g, d, options=None):
    """
    Wolfe line search
    
    Args:
        fun_obj: objective function
        x: current point
        f: current function value
        g: current gradient
        d: search direction
        options: line search options
    
    Returns:
        alpha: step size
        f_new: new function value
        g_new: new gradient
    """
    if options is None:
        options = {}
    
    c1 = options.get('c1', 1e-4)
    c2 = options.get('c2', 0.9)
    
    # use scipy's line search
    alpha, f_new, g_new, _ = line_search(fun_obj, lambda x: fun_obj(x), x, d, 
                                        c1=c1, c2=c2)
    
    if alpha is None:
        alpha = 1.0
        f_new = fun_obj(x + alpha * d)
        g_new = None
    
    return alpha, f_new, g_new

def lbfgs_update(s, y, H0=None, m=10):
    """
    L-BFGS update
    
    Args:
        s: step differences
        y: gradient differences
        H0: initial Hessian approximation
        m: memory size
    
    Returns:
        H: updated Hessian approximation
    """
    if H0 is None:
        H0 = np.eye(len(s))
    
    # simple solution - just returning identity
    # full L-BFGS would be more complex
    return H0

def conjugate_gradient(fun_obj, x0, options=None):
    """
    Conjugate gradient method
    
    Args:
        fun_obj: objective function
        x0: initial point
        options: options
    
    Returns:
        x: optimal point
        f: function value
    """
    if options is None:
        options = {}
    
    max_iter = options.get('MaxIter', 1000)
    tol = options.get('TolFun', 1e-6)
    
    x = x0.copy()
    g = fun_obj(x)
    d = -g
    
    for i in range(max_iter):
        # line search
        alpha, f_new, g_new = wolfe_line_search(fun_obj, x, fun_obj(x), g, d)
        
        if alpha is None or alpha <= 0:
            break
        
        x_new = x + alpha * d
        g_new = fun_obj(x_new)
        
        # check convergence
        if np.linalg.norm(g_new) < tol:
            break
        
        # update direction
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        d = -g_new + beta * d
        
        x = x_new
        g = g_new
    
    return x, fun_obj(x)

def newton_method(fun_obj, x0, hess_fun=None, options=None):
    """
    Newton's method
    
    Args:
        fun_obj: objective function
        x0: initial point
        hess_fun: Hessian function
        options: options
    
    Returns:
        x: optimal point
        f: function value
    """
    if options is None:
        options = {}
    
    max_iter = options.get('MaxIter', 1000)
    tol = options.get('TolFun', 1e-6)
    
    x = x0.copy()
    
    for i in range(max_iter):
        g = fun_obj(x)
        if hess_fun is not None:
            H = hess_fun(x)
            d = -np.linalg.solve(H, g)
        else:
            # use gradient descent if no Hessian
            d = -g
        
        # line search
        alpha, f_new, g_new = wolfe_line_search(fun_obj, x, fun_obj(x), g, d)
        
        if alpha is None or alpha <= 0:
            break
        
        x_new = x + alpha * d
        
        # check convergence
        if np.linalg.norm(g) < tol:
            break
        
        x = x_new
    
    return x, fun_obj(x)
