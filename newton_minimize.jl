using ForwardDiff
using LinearAlgebra
using Distributions
using Random
using CSV
using DataFrames
include("kinetik1.jl")

function line_step_search(x, dir; alpha=1.0)
    global is_gradient_descent = true
    for i in 1:50
        x_new = x + alpha * dir
        if f(x_new) < f(x)
            x = x_new
            break
        elseif i == 50
            is_gradient_descent = false
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

function remove_zeros(v::AbstractVector)
    return filter(x -> x != 0, v)
end

function opt_alg(f::Function, bounds; max_iter=1000)

    # Generate Latin hypercube samples in the search space
    x_samples = latin_hypercube(100, bounds)
    x_min = x_samples[1, :]
    f_min = f(x_min)

    # initiate varible for iteration and sample number
    iter = 0
    sample_num = 0
    for x in eachrow(x_samples)

        # initiate lists for logging results
        sample_num_list::Vector{Int64} = zeros(max_iter)
        x_current_sample_list::Vector{Union{Float64,AbstractArray}} = zeros(max_iter)
        x_current_iter::Vector{Union{Float64,AbstractArray}} = zeros(max_iter)
        function_values = zeros(max_iter)
        term_criteria::Vector{Union{Float64,AbstractArray}} = zeros(max_iter)
        term_reason::Vector{Union{Float64,String}} = zeros(max_iter)

        sample_num += 1
        x_current_samplepoint = x
        for i in 1:max_iter
            # increment interation number used for printing current iteration number
            iter += 1

            # print sample number and interation number 
            println("Iteration number: ", iter, ", Sample number: ", sample_num)

            # Evaluate the function and its gradient and Hessian at the current point
            grad = ForwardDiff.gradient(f, x)
            hess = ForwardDiff.hessian(f, x)

            # To compare with the current x in termination criteria 
            x_prev = x

            # Depending on if the hessian is positive definite or not, either newton or steepest descent is used
            if isposdef(hess)
                x = newton_method(grad, hess, x)
            else
                x = steepest_descent(grad, x)
            end

            # Finite termination criteria            
            eps_1 = 10^-3
            eps_2 = 10^-3
            eps_3 = 10^-3

            # calculate function value used in termination criteria
            function_value = f(x)

            current_term_criteria = []
            # termination criteria 1
            if norm(grad) <= eps_1 * (1 + abs(function_value))
                push!(current_term_criteria, "1")
            end
            # termination criteria 2
            if f(x_prev) - function_value <= eps_2 * (1 + abs(function_value))
                push!(current_term_criteria, "2")
            end
            # termination criteria 3
            if norm(x_prev - x) <= eps_3 * (1 + norm(x))
                push!(current_term_criteria, "3")
            end

            # logging
            sample_num_list[i] = sample_num
            x_current_sample_list[i] = x_current_samplepoint
            x_current_iter[i] = x_prev
            function_values[i] = f(x_prev)
            term_criteria[i] = current_term_criteria

            if !is_gradient_descent
                println("Decent direction not found. Moving on to the next sample point.")
                term_reason[i] = "Decent direction not found"
                break
            elseif length(current_term_criteria) >= 2
                term_reason[i] = "Two or more termination criteria was met"
                break
            else
                term_reason[i] = " "
            end
        end

        data = DataFrame(Samplepoint=remove_zeros(sample_num_list),
            Currentsample=remove_zeros(x_current_sample_list),
            Iteration=remove_zeros(x_current_iter),
            Functionvalues=remove_zeros(function_values),
            Terminationcriteria=remove_zeros(term_criteria),
            Terminationreason=remove_zeros(term_reason))

        # modifying the content of myfile.csv using write method
        CSV.write("data.csv", data; append=true)

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
f(x) = cost_function(problem_object, x, experimental_data)

# Define bounds
bounds = [(0, 11), (0, 11), (0, 11), (0, 11)]

# Find the minimum point and value among the samples
min_point, min_val = opt_alg(f, bounds)

# Print the results
println("Minimum point: ", min_point)
println("Minimum value: ", min_val)
