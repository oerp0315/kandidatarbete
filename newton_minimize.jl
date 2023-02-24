using ForwardDiff
using LinearAlgebra
using Distributions
using Random
using OrdinaryDiffEq

function line_step_search(x, dir; alpha=1)
    for i in 1:50
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

function latin_hypercube(n, bounds, seed=123)

    Random.seed!(seed)

    d = length(bounds)
    samples = zeros(n, d)
    for j in 1:d
        p = rand(1:n, n)
        for i in 1:n
            samples[i, j] = (p[i] - rand()) / n * (bounds[j][2] - bounds[j][1]) + bounds[j][1]
        end
    end
    return samples
end

function opt_alg(f::Function, bounds; tol=1e-6, max_iter=1000)

    # Generate Latin hypercube samples in the search space
    x_samples = latin_hypercube(100, bounds)

    x_min = x_samples[1, :]
    f_min = f(x_min)

    for x in eachrow(x_samples)
        iter = 0 #checking iteration
        x_start = x #jused only to log
        for i in 1:max_iter

            iter += 1 #used only to log
            println("start value: ", x_start, ", x: ", x, ", f(x): ", f(x),  ", iteration: ", iter)

            # Evaluate the function and its gradient and Hessian at the current point
            grad = ForwardDiff.gradient(f, x)
            hess = ForwardDiff.hessian(f, x)

            # Check for convergence
            if norm(grad) < tol
                break
            end
            
            # To compare with the current x in termination criteria 
            x_prev = x

            # Depending on if the hessian is positive definite or not, either newton or steepest descent is used
            if isposdef(hess)
                x = newton_method(grad, hess, x)
            else
                x = steepest_descent(grad, x)
            end

            # Finite termination criteria

            # norm of gradient of f(x) is <= epsilon1*(1+ abs(f(x))
            # f(x_(k-1)) - f(x) <= epsilon2*(1+abs(f(x)))
            # norm of [x_(k-1) - x_k] <= epsilon3*(1+norm of x_k)

            a = 0
            eps_1 = 10^-3
            eps_2 = 10^-3
            eps_3 = 10^-3

            # termination criteria 1
            if norm(ForwardDiff.gradient(f, x)) <= eps_1
                a += 1
                #println(x, " : ", a)
                
            end
            # termination criteria 2
            if f(x) - f(x_prev) <= eps_2*(1+abs(f(x)))
                a += 1
                #println(x, " : ", a)
            end
            # termination criteria 3
            if norm(x_prev - x) <= eps_3*(1 + norm(x))
                a += 1
                #println(x, " : ", a)
            end

            if a >= 2
                break
            end
        
         end

        # Update the minimum point and value
        f_val = f(x)
        if f_val < f_min
            x_min = x
            f_min = f_val
        end
    end

    return x_min, f_min
end

# Define the function to optimize
f(x) = 4 * x[1]^2 - 2.1 * x[1]^4 + (1 / 3) * x[1]^6 + x[1] * x[2] - 4 * x[2]^2 + 4 * x[2]^4

# Generate Latin hypercube samples
n = 10
bounds = [(-1, 1), (-1, 1)]

# Find the minimum point and value among the samples
min_point, min_val = opt_alg(f, bounds)

# Print the results
println("Minimum point: ", min_point)
println("Minimum value: ", min_val)

