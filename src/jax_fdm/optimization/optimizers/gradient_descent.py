from collections.abc import Callable
from typing import Any

import numpy as np
from jaxtyping import Float
from scipy.optimize import OptimizeResult
from scipy.optimize import approx_fprime

from jax_fdm.optimization.optimizers.optimizer import Optimizer
from jax_fdm.optimization.optimizers.optimizer import OptProblem

# ==========================================================================
# Gradient descent
# ==========================================================================


class GradientDescent(Optimizer):
    """
    A vanilla gradient descent optimizer with a fixed learning rate.

    Parameters
    ----------
    learning_rate :
        The step size applied to the negative gradient at each iteration.
    """

    name = "gradient-descent"

    def __init__(self, learning_rate: float = 0.01, **kwargs: Any):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

    def _minimize(self, opt_problem: OptProblem) -> OptimizeResult:
        """
        Dispatch the problem to the in-house gradient descent routine.

        Parameters
        ----------
        opt_problem :
            The problem to minimize.

        Returns
        -------
        result :
            The optimization result.
        """
        opt_problem.options["learning_rate"] = self.learning_rate

        return gradient_descent(**opt_problem.to_kwargs())


# ==========================================================================
# Functions
# ==========================================================================


def gradient_descent(
    fun: Callable,
    x0: Float[np.ndarray, "parameters"],
    args: tuple[Any, ...] = (),
    jac: Callable | bool | None = None,
    tol: float | None = None,
    callback: Callable | None = None,
    options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizeResult:
    """
    Minimize a function by vanilla gradient descent, SciPy-style.

    Parameters
    ----------
    fun :
        The objective to minimize, called as ``fun(x, *args)``. Returns ``(f, grad)``
        when ``jac`` is True, otherwise the scalar objective.
    x0 :
        The initial guess.
    args :
        Extra positional arguments passed to ``fun`` and ``jac``.
    jac :
        The gradient of ``fun``, called as ``jac(x, *args)``. If True, ``fun`` itself
        returns the gradient; if None, a finite-difference gradient is used.
    tol :
        The gradient infinity-norm tolerance for termination.
    callback :
        A function called after each accepted step as ``callback(xk)``.
    options :
        Solver options: ``maxiter`` (default 1000), ``learning_rate`` (default 1e-2),
        ``fd_step`` (finite-difference step, default 1e-8), and the ``ftol`` and
        ``fun_tol`` stopping thresholds.

    Returns
    -------
    result :
        A SciPy-like result carrying the solution, objective, gradient, iteration
        and evaluation counts, and convergence status.

    Notes
    -----
    Iteration stops on any of four conditions: gradient norm below ``tol``, objective
    below ``fun_tol``, a plateau where the objective change falls below ``ftol``, or
    reaching ``maxiter``.
    """
    if options is None:
        options = {}

    maxiter = options.get("maxiter", 1000)
    lr = options.get("learning_rate", 1e-2)
    fd_step = options.get("fd_step", 1e-8)
    ftol = options.get("ftol", tol)
    fun_tol = options.get("fun_tol", tol)

    print(
        f"Gradient descent options: maxiter={maxiter}, lr={lr}, fd_step={fd_step}, "
        f"ftol={ftol}, fun_tol={fun_tol}",
    )

    x = np.asarray(x0)
    nfev = 0
    njev = 0

    # Consolidate function and gradient evaluation
    if jac is True:
        # fun returns (f, grad)
        def eval_fun_and_grad(x_local):  # pyright: ignore[reportRedeclaration]
            nonlocal nfev, njev
            nfev += 1
            njev += 1
            return fun(x_local, *args)

    elif callable(jac):
        # fun returns f, jac returns grad
        def eval_fun_and_grad(x_local):  # pyright: ignore[reportRedeclaration]
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

            # Then compute gradient with approx_fprime (this will call fun multiple
            # times)
            def fun_wrapped(z):
                nonlocal nfev
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
        # scipy's approx_fprime stub resolves to an unrelated sparse-linalg overload
        # (LinearOperator/csr_array union) here;
        # at runtime it always returns a dense ndarray gradient
        grad_norm = np.linalg.norm(grad, ord=np.inf)  # pyright: ignore[reportArgumentType]

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
