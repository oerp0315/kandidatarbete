using ForwardDiff
using FiniteDifferences
using LinearAlgebra
using Distributions
using Random
using CSV
using DataFrames
using DelimitedFiles
include("kinetics_calculator.jl")

function line_step_search(f::Function, x, dir; alpha=1.0)
    is_descent_direction::Bool = true
    for i in 1:50
        x_new = x + alpha * dir
        if f(x_new) < f(x)
            x = x_new
            break
        elseif i == 50
            is_descent_direction = false
        end
        alpha = alpha / 2
    end

    return alpha, is_descent_direction
end

function newton_method(f::Function, grad, hess, x)
    dir = -hess \ grad
    alpha, is_descent_direction = line_step_search(f, x, dir)

    if is_descent_direction
        x += alpha * dir
    end

    return x, is_descent_direction
end

function steepest_descent(f::Function, grad, x)
    dir = -grad / norm(grad)
    alpha, is_descent_direction = line_step_search(f, x, dir)

    if is_descent_direction
        x += alpha * dir
    end

    return x, is_descent_direction
end

function latin_hypercube(n_samples, bounds; seed=123)
    Random.seed!(seed)

    n_vars = length(bounds)


    if isfile("latin_hypercube.csv") && length(readdlm("latin_hypercube.csv", Float64)[1, :]) == n_vars &&
       length(readdlm("latin_hypercube.csv", Float64)[:, 1]) == n_samples

        return readdlm("latin_hypercube.csv", Float64)
    else
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

        open("latin_hypercube.csv", "w") do io
            writedlm(io, samples)
        end

        return samples
    end
end

function remove_zeros(v::AbstractVector)
    return filter(x -> x != 0, v)
end

function check_gradient(f::Function, x)
    # gradient of first sample using ForwardDiff
    grad_forwarddiff = ForwardDiff.gradient(f, x)

    # gradient of first sample using FiniteDifferences
    grad_finitdiff = grad(central_fdm(10, 1), f, x)[1]

    if any(abs.(grad_forwarddiff - grad_finitdiff) .> 1e-3)
        println("Gradient too unstable")
        exit(1)
    end
end

struct log_results
    sample_num_list::Vector{Int64}
    x_current_sample_list::Vector{Union{Float64,AbstractArray}}
    x_current_iter::Vector{Union{Float64,AbstractArray}}
    function_values::Vector{Float64}
    term_criteria::Vector{Union{Float64,AbstractArray,String}}
    term_reason::Vector{Union{Float64,String}}
    cond_num_list::AbstractVector{Union{Float64,String}}
    time_log::Vector{Float64}
end

function opt(f::Function, x, sample_num, iter; max_iter=1000)
    # initiate lists for logging results
    sample_num_list::Vector{Int64} = zeros(max_iter + 1)
    x_current_sample_list::Vector{Union{Float64,AbstractArray}} = zeros(max_iter + 1)
    x_current_iter::Vector{Union{Float64,AbstractArray}} = zeros(max_iter + 1)
    function_values::Vector{Float64} = zeros(max_iter + 1)
    term_criteria::Vector{Union{Float64,AbstractArray,String}} = zeros(max_iter + 1)
    term_reason::Vector{Union{Float64,String}} = zeros(max_iter + 1)
    cond_num_list::AbstractVector{Union{Float64,String}} = zeros(max_iter + 1)
    time_log::Vector{Float64} = zeros(1)

    x_current_samplepoint = x

    # calculate hessian etc. for first point
    grad = ForwardDiff.gradient(f, x)
    hess = ForwardDiff.hessian(f, x)
    func_val = f(x)

    # logging for first x
    sample_num_list[1] = sample_num
    x_current_sample_list[1] = x_current_samplepoint
    x_current_iter[1] = x
    function_values[1] = func_val
    term_criteria[1] = "start point, no termination criteria"
    cond_num_list[1] = "start point, no condition number"
    term_reason[1] = "start point, no reason for temination"

    time = @elapsed for i in 1:max_iter
        # increment interation number used for printing current iteration number
        iter += 1

        # print sample number and interation number 
        println("Iteration number: ", iter, ", Sample number: ", sample_num)

        # Evaluate the function and its gradient and Hessian at the current point
        grad = ForwardDiff.gradient(f, x)
        hess = ForwardDiff.hessian(f, x)

        # calculate condtion number
        cond_num = cond(hess)

        # To compare with the current x in termination criteria 
        x_prev = x

        is_descent_direction::Bool = false

        # Depending on if the hessian is positive definite or not, either newton or steepest descent is used
        if isposdef(hess)
            x, is_descent_direction = newton_method(f, grad, hess, x)
        end

        if !is_descent_direction
            x, is_descent_direction = steepest_descent(f, grad, x)
        end

        if !is_descent_direction
            println("Descent direction not found!")
            break
        end

        # Finite termination criteria            
        eps = 1e-3

        # calculate function value used in termination criteria
        function_value = f(x)

        current_term_criteria = []
        # termination criteria 1
        if norm(grad) <= eps * (1 + abs(function_value))
            push!(current_term_criteria, "1")
        end
        # termination criteria 2
        if f(x_prev) - function_value <= eps * (1 + abs(function_value))
            push!(current_term_criteria, "2")
        end
        # termination criteria 3
        if norm(x_prev - x) <= eps * (1 + norm(x))
            push!(current_term_criteria, "3")
        end

        # logging
        sample_num_list[i+1] = sample_num
        x_current_sample_list[i+1] = x_current_samplepoint
        x_current_iter[i+1] = x
        function_values[i+1] = f(x)
        term_criteria[i+1] = current_term_criteria
        cond_num_list[i+1] = cond_num

        if length(current_term_criteria) >= 2
            term_reason[i+1] = "Two or more termination criteria was met"
            break
        else
            term_reason[i+1] = " "
        end
    end

    # log time for each sample
    time_log[1] = time

    res = log_results(remove_zeros(sample_num_list),
        remove_zeros(x_current_sample_list),
        remove_zeros(x_current_iter),
        remove_zeros(function_values),
        remove_zeros(term_criteria),
        remove_zeros(term_reason),
        remove_zeros(cond_num_list),
        time_log)

    return res, iter, x
end

function p_est(f::Function, bounds, n_samples)
    # Check if the data.csv exists and truncate it if it does
    data_file = open("data.csv", "w")
    if isfile("data.csv")
        truncate(data_file, 0)
    end
    close(data_file)

    # Check if the time_log.csv exists and truncate it if it does
    timelog_file = open("time_log.csv", "w")
    if isfile("time_log.csv")
        truncate(timelog_file, 0)
    end
    close(timelog_file)

    # Generate Latin hypercube samples in the search space
    x_samples = latin_hypercube(n_samples, bounds)

    # if the gradient is not good enough the program will terminate
    check_gradient(f, x_samples[1, :])

    x_min = x_samples[1, :]
    f_min = f(x_min)
    iter_min = 1

    # initiate varible sample and iteration number
    sample_num = 0
    iter = 0

    for x in eachrow(x_samples)
        sample_num += 1

        # minimizes the cost function for the current start-guess
        res, iter, x = opt(f::Function, x, sample_num, iter)

        data = DataFrame(Samplepoint=res.sample_num_list,
            Currentsample=res.x_current_sample_list,
            Iteration=res.x_current_iter,
            Functionvalues=res.function_values,
            Condnum=res.cond_num_list,
            Terminationcriteria=res.term_criteria,
            Terminationreason=res.term_reason)

        # modifying the content of myfile.csv using write method
        CSV.write("data.csv", data; append=true)

        # log time for each sample point
        CSV.write("time_log.csv", DataFrame(time=res.time_log); append=true)

        # Update the minimum point and value
        f_val = f(x)
        if f_val < f_min
            x_min = x
            f_min = f_val
            iter_min = iter
        end
    end

    # Print the results
    println("Minimum point: ", x_min)
    println("Minimum value: ", f_min)
    println("Iteration: ", iter_min)

    return x_min, f_min
end



# Define the function to optimize
f(x) = cost_function(problem_object, x, experimental_data)

function intermediate_cost_function(x_small, index_x_small, x_big)
    _x_big = convert.(eltype(x_small), x_big)
    _x_big[index_x_small] .= x_small
    return cost_function(problem_object, _x_big, experimental_data)
end

# Omdefinera när det behövs
cost_function_profilelikelihood = (x) -> intermediate_cost_function(x, [1, 2, 4], x_hatt)
# Define bounds
bounds = [(0, 11), (0, 11), (0, 11)]

p_est(f, bounds, 500)
