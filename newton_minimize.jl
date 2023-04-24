using ForwardDiff
using FiniteDifferences
using LinearAlgebra
using Distributions
using Random
using CSV
using DataFrames
using DelimitedFiles
include("model_easy.jl")

"Search for the step size in the gradient direction dir used to find the next point for function f at point x"
function line_step_search(f::Function, x, dir; alpha=1.0)
    is_descent_direction::Bool = true

    # divide the start step size (alpha) until a function value less than that of the previous point is found
    for i in 1:50
        x_new = x + alpha * dir
        if f(x_new) == Inf
            alpha /= 2
            continue
        elseif f(x_new) < f(x) && i != 50
            break
        elseif i == 50
            is_descent_direction = false
        end

        alpha /= 2
    end

    return alpha, is_descent_direction
end

#Newton method used to find a point in a descending direction
function newton_method(f::Function, grad, hess, x, log_bounds)
    dir = -hess \ grad
    # obtain step size alpha
    alpha, is_descent_direction = line_step_search(f, x, dir)

    # calculate next point and ensure point is within bounds, project back if that is the case
    if is_descent_direction
        x += alpha * dir
        for i in eachindex(log_bounds)
            if x[i] < log_bounds[i][1]
                x[i] = log_bounds[i][1]
            elseif x[i] > log_bounds[i][2]
                x[i] = log_bounds[i][2]
            end
        end
    end

    return x, is_descent_direction
end

#Alternative method to find a point in a descending direction
function steepest_descent(f::Function, grad, x, log_bounds)
    dir = -grad / norm(grad)
    alpha, is_descent_direction = line_step_search(f, x, dir)

    # calculate next point and ensure point is within bounds, project back if that is the case
    if is_descent_direction
        x += alpha * dir
        for i in eachindex(log_bounds)
            if x[i] < log_bounds[i][1]
                x[i] = log_bounds[i][1]
            elseif x[i] > log_bounds[i][2]
                x[i] = log_bounds[i][2]
            end
        end
    end

    return x, is_descent_direction
end

"Generate samples according to latin square method with same dimentions as bounds"
function latin_hypercube(n_samples, bounds; seed=123)
    # fix seed
    Random.seed!(seed)

    # make a folder for the result of parameter estimation
    if isdir("p_est_results") == false
        mkdir("p_est_results")
    end

    n_vars = length(bounds)

    # previous bounds
    if isfile("p_est_results/bounds.csv")
        read_previous_bounds = CSV.File("p_est_results/bounds.csv") |> DataFrame
        previous_bounds = [(x, y) for (x, y) in zip(read_previous_bounds[:, 1], read_previous_bounds[:, 2])]
    end

    # if identical settings as previously, use the same samples. Otherwise create new samples
    if isfile("p_est_results/bounds.csv") && bounds == previous_bounds &&
       length(readdlm("p_est_results/latin_hypercube.csv", Float64)[:, 1]) == n_samples

        return readdlm("p_est_results/latin_hypercube.csv", Float64)
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

        # save generated samples in file
        open("p_est_results/latin_hypercube.csv", "w") do io
            writedlm(io, samples)
        end

        # log used bounds
        CSV.write("p_est_results/bounds.csv", DataFrame(bounds))

        return samples
    end
end

"Remove elements in a vector equal to zeros"
function remove_zeros(v::AbstractVector)
    return filter(x -> x != 0, v)
end

"Check the quality of a gradient of a function f at x with ForwardDiff in comparison to FiniteDifferences"
function check_gradient(f::Function, x)
    # gradient of first sample using ForwardDiff
    grad_forwarddiff = ForwardDiff.gradient(f, x)

    # gradient of first sample using FiniteDifferences
    grad_finitdiff = grad(central_fdm(10, 1), f, x)[1]

    # if gradient differs more than a tolerance the gradient is not good enough and the code stops
    if any(abs.(grad_forwarddiff - grad_finitdiff) .> 1e-3)
        println("Gradient too unstable")
        exit(1)
    end
end

# struct for collecting data for logging
struct log_results
    sample_num_list::Vector{Int64}
    x_current_sample_list::Vector{Union{Float64,AbstractArray}}
    x_current_iter::Vector{Union{Float64,AbstractArray}}
    function_values::Vector{Float64}
    term_criteria::Vector{Union{Float64,AbstractArray,String}}
    term_reason::Vector{Union{Float64,String}}
    descent_method::Vector{Union{Float64,AbstractArray,String}}
    cond_num_list::AbstractVector{Union{Float64,String}}
    time_log::Vector{Float64}
end

"Minimizes a function f a point x with a combination of steepest descent and newtons method"
function opt(f::Function, x, sample_num, iter, log_bounds; max_iter=1000)
    # initiate lists for logging results
    sample_num_list::Vector{Int64} = zeros(max_iter + 1)
    x_current_sample_list::Vector{Union{Float64,AbstractArray}} = zeros(max_iter + 1)
    x_current_iter::Vector{Union{Float64,AbstractArray}} = zeros(max_iter + 1)
    function_values::Vector{Float64} = zeros(max_iter + 1)
    term_criteria::Vector{Union{Float64,AbstractArray,String}} = zeros(max_iter + 1)
    term_reason::Vector{Union{Float64,String}} = zeros(max_iter + 1)
    descent_method::Vector{Union{Float64,AbstractArray,String}} = zeros(max_iter + 1)
    cond_num_list::AbstractVector{Union{Float64,String}} = zeros(max_iter + 1)
    time_log::Vector{Float64} = zeros(1)

    x_current_samplepoint = x

    # calculate hessian etc. for first point
    grad = ForwardDiff.gradient(f, x)
    hess = ForwardDiff.hessian(f, x)
    func_val = f(x)

    # logging for first x
    sample_num_list[1] = sample_num
    x_current_sample_list[1] = exp.(x_current_samplepoint)
    x_current_iter[1] = exp.(x)
    function_values[1] = func_val
    term_criteria[1] = "start point, no termination criteria"
    descent_method[1] = "start point, not descended yet"
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

        # initiate a list for logging the descent method used for an optimization step
        current_descent_method = []

        # Depending on if the hessian is positive definite or not, either newton or steepest descent is used
        if isposdef(hess)
            x, is_descent_direction = newton_method(f, grad, hess, x, log_bounds)
            #logging the used descent method
            push!(current_descent_method, "Newton method")
        end

        if !is_descent_direction
            x, is_descent_direction = steepest_descent(f, grad, x, log_bounds)
            #logging the used descent method
            push!(current_descent_method, "Steepest descent")

        end

        #= if a descent direction could not be found, the optmimization of the current sample is terminated
        and continue with the next =#
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
        x_current_sample_list[i+1] = exp.(x_current_samplepoint)
        x_current_iter[i+1] = exp.(x)
        function_values[i+1] = f(x)
        term_criteria[i+1] = current_term_criteria
        descent_method[i+1] = current_descent_method
        cond_num_list[i+1] = cond_num

        # Checks if two or more of the termination criteria are met
        if length(current_term_criteria) >= 2
            term_reason[i+1] = "Two or more termination criteria was met"
            break
        else
            term_reason[i+1] = " "
        end
    end

    # log time for each sample
    time_log[1] = time

    # create a log_results object for current logging data
    res = log_results(remove_zeros(sample_num_list),
        remove_zeros(x_current_sample_list),
        remove_zeros(x_current_iter),
        remove_zeros(function_values),
        remove_zeros(term_criteria),
        remove_zeros(term_reason),
        remove_zeros(descent_method),
        remove_zeros(cond_num_list),
        time_log)

    return res, iter, x
end

"Runs an optimization on function f in the region of bounds with n_samples number of samples.
If running p_est through profile likelihood pl_mode should be true"
function p_est(f::Function, bounds, n_samples, pl_mode; x_samples_log=0)
    if pl_mode == false
        # create a directory for parameter estimation
        if isdir("p_est_results") == false
            mkdir("p_est_results")
        end
        # Check if the data.csv exists and truncate it if it does
        if isfile("p_est_results/data.csv")
            data_file = open("p_est_results/data.csv", "w")
            truncate(data_file, 0)
            close(data_file)
        end

        # Check if the time_log.csv exists and truncate it if it does
        if isfile("p_est_results/time_log.csv")
            timelog_file = open("p_est_results/time_log.csv", "w")
            truncate(timelog_file, 0)
            close(timelog_file)
        end

        # Generate Latin hypercube samples in the search space
        x_samples = latin_hypercube(n_samples, bounds)

        # logarithmize the samples
        x_samples_log = log.(x_samples)
    end

    # transform bounds to log-scale
    log_bounds = map(x -> (log(x[1]), log(x[2])), bounds)

    # if the gradient is not good enough the program will terminate
    check_gradient(f, x_samples_log[1, :])

    # start values, set for first sample
    x_min = x_samples_log[1, :]
    f_min = f(x_min)

    # initiate varable that holds the iteration that has given the lowest function value
    iter_min = 1

    # initiate varible sample and iteration number
    sample_num = 0
    iter = 0

    # iterate over the samples, each sample is optimized 
    for x in eachrow(x_samples_log)
        sample_num += 1

        # minimizes the cost function for the current start-guess
        res, iter, x = opt(f::Function, x, sample_num, iter, log_bounds)

        # only necessary if Profile likelihood is not currently used
        if pl_mode == false
            data = DataFrame(Samplepoint=res.sample_num_list,
                Currentsample=res.x_current_sample_list,
                Iteration=res.x_current_iter,
                Functionvalues=res.function_values,
                Condnum=res.cond_num_list,
                Terminationcriteria=res.term_criteria,
                Descentmethod=res.descent_method,
                Terminationreason=res.term_reason)

            # modifying the content of data.csv using write method
            CSV.write("p_est_results/data.csv", data; append=true)

            # log time for each sample point
            CSV.write("p_est_results/time_log.csv", DataFrame(time=res.time_log); append=true)
        end

        # Update the minimum point and value
        f_val = f(x)
        if f_val < f_min
            x_min = exp.(x)
            f_min = f_val
            iter_min = iter
        end
    end

    # Print the results
    println("Minimum point: ", x_min)
    println("Minimum value: ", f_min)
    println("Iteration resposible for minimum: ", iter_min)

    return x_min, f_min
end

# Define the function to optimize
f(x) = cost_function(problem_object, x, experimental_data)

# Define bounds
bounds = [(0.1, 6), (0.1, 6)]

# run the parameter estimation
x_min, f_min = p_est(f, bounds, 10, false)
