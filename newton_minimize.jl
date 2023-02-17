using ForwardDiff
using LinearAlgebra

function line_step_search(x, dx; alpha=0.1)
    for i in 1:10
        x_new = x + alpha * dx
        if f(x_new) < f(x)
            x = x_new
            break
        end
        alpha = alpha / 2
    end

    return alpha
end

function newton_method(gx, Hx, x)
    dx = -Hx \ gx
    x += line_step_search(x, dx) * dx

    return x
end

function steepest_descent(grad, x)
    dx = -grad / norm(grad)
    x += line_step_search(x, dx) * dx

    return x
end

function opt_alg(f, x0; tol=1e-6, max_iter=100)
    x = x0
    for i in 1:max_iter
        # Evaluate the function and its gradient and Hessian at the current point
        grad = ForwardDiff.gradient(f, x)
        hess = ForwardDiff.hessian(f, x)

        # Check for convergence
        if norm(grad) < tol
            break
        end

        # Depending on if the hessian is positive definite or not, either newton or steepest descent is used
        if isposdef(hess)
            x = newton_method(grad, hess, x)
        else
            x = steepest_descent(grad, x)
        end
    end

    return x
end

f(x) = sin(x[1]) + sin(x[2])

println(opt_alg(f, [1, 2]))

rand()
