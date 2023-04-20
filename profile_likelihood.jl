include("newton_minimize.jl")

function new_point(log_param_last, log_params, param_index, log_bounds, sign, threshold; abstol=1e-6, reltol=1e-2)
    stop_flag = false
    step_size = zeros(length(log_param_last))
    step_size[param_index] = log_param_last[param_index]
    if step_size[param_index] < 0
        step_size[param_index] = -step_size[param_index]
    end
    cond_val = abs(f(log_param_last + sign * step_size) - f(log_param_last))

    if f(log_param_last + sign * step_size) == Inf || cond_val > abstol + reltol * f(log_param_last)
        while f(log_param_last + sign * step_size) == Inf || cond_val > abstol + reltol * f(log_param_last)
            step_size[param_index] /= 2
            if step_size[param_index] < 1e-6
                step_size[param_index] = 1e-6
                stop_flag = true
                break
            end
            cond_val = abs(f(log_param_last + sign * step_size) - f(log_param_last))
        end
    elseif cond_val <= abstol + reltol * f(log_param_last)
        while cond_val <= abstol + reltol * f(log_param_last)
            step_size[param_index] *= 2
            if step_size[param_index] > 2.0 * log_param_last[param_index]
                step_size[param_index] = 2.0 * log_param_last[param_index]
                stop_flag = true
                break
            end
            cond_val = abs(f(log_param_last + sign * step_size) - f(log_param_last))
            if cond_val > abstol + reltol * f(log_param_last)
                step_size[param_index] /= 2
                break
            end
        end
    end
    new_point = log_param_last + sign * step_size

    # point can not be outside of bounds
    if sign == -1
        if new_point[param_index] < log_bounds[param_index][1]
            new_point[param_index] = log_bounds[param_index][1]
            stop_flag = true
        end
    elseif sign == 1
        if new_point[param_index] > log_bounds[param_index][2]
            new_point[param_index] = log_bounds[param_index][2]
            stop_flag = true
        end
    end

    return new_point, stop_flag
end

function intermediate_cost_function(x_small, index_x_small, x_big)
    x_big_ = convert.(eltype(x_small), x_big)
    x_big_[index_x_small] .= x_small

    return cost_function(problem_object, x_big_, experimental_data)
end

struct log_pl_results
    fix_param_index::Vector{Int64}
    fix_param_list::Vector{Float64}
    x_list::Vector{Union{Float64,AbstractArray}}
    costfunc_value_list::Vector{Float64}
end

# Define the function to perform profile likelihood analysis for one parameter
function profile_likelihood(params, param_index, bounds, num_points, threshold)
    # list of indexes to be optimized
    index_list = [i for i in 1:length(bounds) if i != param_index]

    # log-scale parameters
    log_params = log.(params)

    # log-scale bounds
    log_bounds = map(x -> (log(x[1]), log(x[2])), bounds)

    # new bounds
    bounds_ = copy(bounds)
    current_bounds = deleteat!(bounds_, param_index)

    # new start values
    x_samples = readdlm("p_est_results/latin_hypercube.csv", Float64)

    new_x_samples = hcat(x_samples[:, 1:param_index-1], x_samples[:, param_index+1:end])

    new_x_samples_log = log.(new_x_samples)

    stop_flag = false
    sign = -1

    # initiate lists for logging
    fix_param_index::Vector{Int64} = zeros(2 * num_points + 1)
    fix_param_list::Vector{Union{Float64,AbstractArray}} = zeros(2 * num_points + 1)
    x_list::Vector{Union{Float64,AbstractArray}} = zeros(2 * num_points + 1)
    costfunc_value_list::Vector{Float64} = zeros(2 * num_points + 1)

    # log optimized parameters (start values)
    fix_param_index[Int(num_points)+1] = param_index
    fix_param_list[Int(num_points)+1] = x_min[param_index]
    x_list[Int(num_points)+1] = [x_min[i] for i in eachindex(x_min) if i != param_index]
    costfunc_value_list[Int(num_points)+1] = f_min

    i = 0
    log_params_current = log_params

    while i < num_points
        i += 1

        if stop_flag == false && i != num_points
            # calculate next point
            log_params_current, stop_flag = new_point(log_params_current, log_params, param_index, log_bounds, sign, threshold)

        elseif (i == num_points && sign == -1 && !(stop_flag == true)) || (!(i == num_points) && sign == -1 && stop_flag == true)  # kanske behöver kollas över
            sign = 1
            i = 1
            log_params_current = log_params
            log_params_current, stop_flag = new_point(log_params_current, log_params, param_index, log_bounds, sign, threshold)
        else
            break
        end

        # Omdefinera kostfuntionen
        cost_function_profilelikelihood = (x) -> intermediate_cost_function(x, index_list, log_params_current)

        # Find the maximum likelihood estimate for the parameter of interest
        x_min, f_min = p_est(cost_function_profilelikelihood, current_bounds, 10, true, new_x_samples_log)

        if sign == -1
            fix_param_index[Int(num_points)+1-i] = param_index
            fix_param_list[Int(num_points)+1-i] = exp.(log_params_current[param_index])
            x_list[Int(num_points)+1-i] = x_min
            costfunc_value_list[Int(num_points)+1-i] = f_min
        else
            fix_param_index[Int(num_points)+1+i] = param_index
            fix_param_list[Int(num_points)+1+i] = exp.(log_params_current[param_index])
            x_list[Int(num_points)+1+i] = x_min
            costfunc_value_list[Int(num_points)+1+i] = f_min
        end

        if f_min > f(log_params) * 1.2
            stop_flag = true
        end
    end

    # puts data in struct and removes zeros
    pl_res = log_pl_results(remove_zeros(fix_param_index),
        remove_zeros(fix_param_list),
        remove_zeros(x_list),
        remove_zeros(costfunc_value_list))

    return pl_res
end

function run_profile_likelihood(params, bounds, num_points, threshold)
    # create a directory for profile likelihood
    if isdir("profilelikelihood_results") == false
        mkdir("profilelikelihood_results")
    end

    # Check if the profile_likelihood.csv exists and truncate it if it does
    if isfile("profilelikelihood_results/profile_likelihood.csv")
        pl_file = open("profilelikelihood_results/profile_likelihood.csv", "w")
        truncate(pl_file, 0)
        close(pl_file)
    end

    for i in 1:length(bounds)
        pl_res = profile_likelihood(params, i, bounds, num_points, threshold)

        data = DataFrame(Fixed_parameter_index=pl_res.fix_param_index,
            Fixed_parameter=pl_res.fix_param_list,
            Parameters=pl_res.x_list,
            CostfunctionValues=pl_res.costfunc_value_list)

        # modifying the content of profile_likelihood.csv using write method
        CSV.write("profilelikelihood_results/profile_likelihood.csv", data; append=true)

        #data to plot
        fixed_parameter = pl_res.fix_param_list
        costfunction_values = pl_res.costfunc_value_list

        #plot
        plot(fixed_parameter, costfunction_values, xaxis="Parameter $i", yaxis="Cost function values")

        #save plot
        savefig("profilelikelihood_results/parameter$i.png")
    end
end

function contourplot_2p()
    x = collect(range(0.1, 6, length=100))
    y = collect(range(0.1, 5, length=100))
    points = Vector{Vector{Float64}}(undef, length(x) * length(y))

    k = 1
    for i in x
        for j in y
            points[k] = [i, j]
            k += 1
        end
    end

    A = zeros(100, 100)
    index = 1 # initialize index for vector
    for i in 1:100
        for j in 1:100
            # Generate the element you want to insert
            element = f(log.(points[index]))

            # Insert the element into the matrix at the current index
            A[i, j] = element

            index += 1 # increment index for next element in vector
        end
    end
    contour(x, y, A, levels=30)
    savefig("plot_2d.png")
end

# Define the initial parameter values
params = x_min

# Perform profile likelihood analysis for each parameter
num_points = 100
threshold = 3.81 # For 95% confidence interval

run_profile_likelihood(params, bounds, num_points, threshold)

#contourplot_2p()
