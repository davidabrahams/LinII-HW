from scipy.optimize import fmin
from numpy import sin, cos, vstack, array, meshgrid, linspace, zeros
import matplotlib.pyplot as plt
import seaborn as sns

def function(x, y):
    return sin(x) * sin(x + 3 * y)

def contour(func, x_min, x_max, y_min, y_max):

    x = linspace(x_min, x_max)
    y = linspace(y_min, y_max)
    X, Y = meshgrid(x, y)

    Z = func(X, Y)

    CS = plt.contour(X, Y, Z)

def grad_f(x, y):
    d_dx = sin(2*x + 3*y)
    d_dy = 3 * sin(x) * cos(x + 3 * y)
    return d_dx, d_dy

def quiver(grad_func, x_min, x_max, y_min, y_max):
    x = linspace(x_min, x_max, num=20)
    y = linspace(y_min, y_max, num=20)
    X, Y = meshgrid(x, y)

    U, V = grad_func(X, Y)
    q = plt.quiver(X, Y, U, V)

def descent(grad_func, x_0, y_0, lambda_val, n_iterations):

    res = vstack((array([x_0, y_0]), zeros((n_iterations, 2))))

    for i in range(1, n_iterations + 1):
        prev_x, prev_y = res[i - 1, :]
        gradient_x, gradient_y = grad_func(prev_x, prev_y)
        next_x = lambda_val * gradient_x + prev_x
        next_y = lambda_val * gradient_y + prev_y
        res[i, :] = [next_x, next_y]

    x = res[:, 0]
    y = res[:, 1]
    q = plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',
               angles='xy', scale=1)
    return res

def accurate_descent(func, grad_func, x_0, y_0, n_iterations):

    res = vstack((array([x_0, y_0]), zeros((n_iterations, 2))))

    for i in range(1, n_iterations + 1):
        prev_x, prev_y = res[i - 1, :]
        gradient_x, gradient_y = grad_func(prev_x, prev_y)
        lambda_val = optimize_lambda(func, prev_x, prev_y, gradient_x,
                                     gradient_y)
        next_x = lambda_val * gradient_x + prev_x
        next_y = lambda_val * gradient_y + prev_y
        res[i, :] = [next_x, next_y]

    x = res[:, 0]
    y = res[:, 1]
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy',
               angles='xy', scale=1)

    return res

def optimize_lambda(func, x_0, y_0, grad_x, grad_y):
    anon_func = lambda x: -f_x_lamba_f(func, x_0, y_0, grad_x, grad_y, x)
    return fmin(anon_func, 0)

def f_x_lamba_f(func, x_0, y_0, grad_x, grad_y, lambda_val):
    next_x = x_0 + grad_x * lambda_val
    next_y = y_0 + grad_y * lambda_val
    return func(next_x, next_y)


def main():
    plt.figure()
    x_min, x_max = 0, 2
    y_min, y_max = -.5, 1.5
    # x_min, x_max = -2, 2
    # y_min, y_max = -2, 2
    start_x, start_y = 1, 1
    lambda_val = 0.1
    iterations = 10

    contour(function, x_min, x_max, y_min, y_max)
    # quiver(grad_f, x_min, x_max, y_min, y_max)
    # descent(grad_f, start_x, start_y, lambda_val, iterations)
    accurate_descent(function, grad_f, start_x, start_y,
                     iterations)
    # plt.title("Contour plot of f(x, y)")
    plt.title("Optimized Gradient Descent of f(x, y).\nStarting at (1, 1), lambda set to maximize f(x, y)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    main()
