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

function latin_hypercube(n_samples, bounds; seed=123)
    Random.seed!(seed)

    if n_samples < 2
        error("n must be at least 2")
    end

    n_vars = length(bounds)

    # Initialize the Latin square as an n-by-n array of zeros
    square = zeros(Int, n_samples, n_samples)

    # Fill the first row with random integers between 1 and n
    square[1, :] = randperm(n_samples)

    # Fill the remaining rows with shifted copies of the first row
    for i in 2:n_samples
        square[i, :] = circshift(square[i-1, :], 1)
    end

    # create random values to be added to sample values
    random_matrix = rand(n_samples, n_vars)

    # create a matrix where samples will be inserted to
    samples = zeros(n_samples, n_vars)

    # generate samples with random position within varible intervals
    for i in 1:n_samples
        for j in 1:n_vars
            samples[i, j] = (square[i, j] - 1) / ((n_samples - 1) * (n_samples / (n_samples - 1))) + random_matrix[i, j] / n_samples
        end
    end

    # scale samples to bounds
    for i in 1:n_samples
        for j in 1:n_vars
            samples[i, j] = (bounds[j][2] - bounds[j][1]) * samples[i, j] + bounds[j][1]
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
            if norm(ForwardDiff.gradient(f, x)) <= eps_1*(1 + abs(f(x))) 
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

# Define bounds
bounds = [(-1, 1), (-1, 1)]

# Find the minimum point and value among the samples
min_point, min_val = opt_alg(f, bounds)

# Print the results
println("Minimum point: ", min_point)
println("Minimum value: ", min_val)
