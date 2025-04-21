"""A module providing numerical solvers for nonlinear equations."""


class ConvergenceError(Exception):
    """Exception raised if a solver fails to converge."""

    pass


def newton_raphson(f, df, x_0, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using Newton-Raphson iteration.

    Solve f==0 using Newton-Raphson iteration.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the iteration.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using Newton iteration.
    """
    x = x_0

    for i in range(max_its):
        if abs(f(x)) < eps:
            return x
        else:
            x = x - f(x)/df(x)

    if abs(f(x)) < eps:
        return x
    else:
        raise ConvergenceError


def bisection(f, x_0, x_1, eps=1.0e-5, max_its=20):
    """Solve a nonlinear equation using bisection.

    Solve f==0 using bisection starting with the interval [x_0, x_1]. f(x_0)
    and f(x_1) must differ in sign.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    x_0 : float
        The left end of the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its : int
        The maximum number of iterations to be taken before the solver is taken
        to have failed.

    Returns
    -------
    float
        The approximate root computed using bisection.
    """
    if f(x_0)*f(x_1) > 0:
        raise ValueError("Expected signs to be different.")

    for i in range(max_its):
        x_star = (x_0 + x_1)/2
        if f(x_0)*f(x_1) > 0:
            raise ValueError("Expected signs to not be the same.")

        if abs(f(x_star)) < eps:
            return x_star

        if f(x_0)*f(x_1) < 0:
            x_0 = x_star
        else:
            x_1 = x_star

    if abs(f(x_star)) < eps:
        return x_star
    else:
        raise ConvergenceError


def solve(f, df, x_0, x_1, eps=1.0e-5, max_its_n=20, max_its_b=20):
    """Solve a nonlinear equation.

    solve f(x) == 0 using Newton-Raphson iteration, falling back to bisection
    if the former fails.

    Parameters
    ----------
    f : function(x: float) -> float
        The function whose root is being found.
    df : function(x: float) -> float
        The derivative of f.
    x_0 : float
        The initial value of x in the Newton-Raphson iteration, and left end of
        the initial bisection interval.
    x_1 : float
        The right end of the initial bisection interval.
    eps : float
        The solver tolerance. Convergence is achieved when abs(f(x)) < eps.
    max_its_n : int
        The maximum number of iterations to be taken before the newton-raphson
        solver is taken to have failed.
    max_its_b : int
        The maximum number of iterations to be taken before the bisection
        solver is taken to have failed.

    Returns
    -------
    float
        The approximate root.
    """
    try:
        newton_raphson(f, df, x_0, x_1, eps, max_its_n)
    except ConvergenceError:
        bisection(f, x_0, x_1, eps, max_its_b)
