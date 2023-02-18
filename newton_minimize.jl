using ForwardDiff
using LinearAlgebra

function line_step_search(x, dir; alpha=1)
    for i in 1:10
        x_new = x + alpha * dir
        if f(x_new) < f(x)
            x = x_new
            break
        end
        alpha = alpha / 2
    end

    return alpha
end

function newton_method(grad, hess, x)
    dir = -hess \ grad
    x += line_step_search(x, dir) * dir

    return x
end

function steepest_descent(grad, x)
    dir = -grad / norm(grad)
    x += line_step_search(x, dir) * dir

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
