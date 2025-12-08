import numpy as np

from scipy.optimize import approx_fprime
from scipy.optimize import OptimizeResult

from jax_fdm.optimization.optimizers import Optimizer


# ==========================================================================
# Gradient descent
# ==========================================================================

class GradientDescent(Optimizer):
    """
    The gradient descent algorithm.
    """
    def __init__(self, learning_rate=0.01, **kwargs):
        super().__init__(name="gradient-descent", **kwargs)
        self.learning_rate = learning_rate

    def _minimize(self, opt_problem):
        """
        Custom backend method to minimize a loss function.
        """
        opt_problem["options"]["learning_rate"] = self.learning_rate

        return gradient_descent(**opt_problem)


# ==========================================================================
# Functions
# ==========================================================================

def gradient_descent(fun, x0, args=(), jac=None, tol=None, callback=None, options=None, **kwargs):
    """
    Simple vanilla gradient descent with a SciPy-like interface.

    Parameters
    ----------
    fun : callable
        Objective function to be minimized.
        Signature: fun(x, *args) -> float
    x0 : array_like
        Initial guess.
    args : tuple, optional
        Extra arguments passed to `fun` and `jac`.
    jac : {callable, None}, optional
        Gradient (Jacobian) of `fun`.
        Signature: jac(x, *args) -> array_like
        If None, SciPy's approx_fprime is used.
    tol : float, optional
        Gradient infinity-norm tolerance for termination.
    callback : callable, optional
        Called after each iteration as callback(xk).
    options : dict, optional
        Options for the solver:
            - 'maxiter' (int): maximum iterations (default: 1000)
            - 'learning_rate' (float): step size (default: 1e-2)
            - 'fd_step' (float): finite-difference step (default: 1e-8)

    Returns
    -------
    result : OptimizeResult
        SciPy-like result object with fields:
        x, fun, jac, nit, nfev, njev, success, status, message
    """
    if options is None:
        options = {}

    maxiter = options.get("maxiter", 1000)
    lr = options.get("learning_rate", 1e-2)
    fd_step = options.get("fd_step", 1e-8)
    ftol = options.get("ftol", tol)
    fun_tol = options.get("fun_tol", tol)

    print(f"Gradient descent options: maxiter={maxiter}, lr={lr}, fd_step={fd_step}, ftol={ftol}, fun_tol={fun_tol}")

    x = np.asarray(x0)
    nfev = 0
    njev = 0

    # Consolidate function and gradient evaluation
    if jac is True:
        # fun returns (f, grad)
        def eval_fun_and_grad(x_local):
            nonlocal nfev, njev
            nfev += 1
            njev += 1
            return fun(x_local, *args)

    elif callable(jac):
        # fun returns f, jac returns grad
        def eval_fun_and_grad(x_local):
            nonlocal nfev, njev
            nfev += 1
            f_val = fun(x_local, *args)
            njev += 1
            g_val = jac(x_local, *args)
            return f_val, g_val

    else:
        # No jacobian provided: use finite differences via approx_fprime
        def eval_fun_and_grad(x_local):
            nonlocal nfev, njev
            x_arr = np.asarray(x_local)

            # First evaluate f at x (once)
            nfev += 1
            f0 = fun(x_arr, *args)

            # Then compute gradient with approx_fprime (this will call fun multiple times)
            def fun_wrapped(z):
                nfev += 1
                return fun(z, *args)

            g_val = approx_fprime(x_arr, fun_wrapped, epsilon=fd_step)
            njev += 1

            return f0, g_val

    # Initialization
    success = False
    status = 1
    message = "Maximum number of iterations reached."

    # Initial function value and gradient
    f_current, grad = eval_fun_and_grad(x)

    success = False
    status = 1
    message = "Maximum number of iterations reached."

    for k in range(maxiter):
        grad_norm = np.linalg.norm(grad, ord=np.inf)

        # Gradient-based stopping
        if tol is not None and grad_norm < tol:
            success = True
            status = 0
            message = "Gradient norm below tolerance."
            break

        # Absolute function value stopping
        if fun_tol is not None and f_current <= fun_tol:
            success = True
            status = 2
            message = "Function value below fun_tol."
            break

        # Gradient descent update
        x_new = x - lr * grad

        if callback is not None:
            callback(x_new)

        f_new, grad_new = eval_fun_and_grad(x_new)

        # 3) Plateau stopping: small change in function value
        if ftol is not None:
            if abs(f_new - f_current) <= ftol * (1.0 + abs(f_current)):
                success = True
                status = 3
                message = "Function value change below ftol (plateau)."
                x, grad, f_current = x_new, grad_new, f_new
                break

        # Accept the step
        x, grad, f_current = x_new, grad_new, f_new

    # Assemble result
    result = OptimizeResult(
        x=x,
        fun=f_current,
        jac=grad,
        nit=k + 1,
        nfev=nfev,
        njev=njev,
        success=success,
        status=status,
        message=message,
    )

    return result
