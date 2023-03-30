using Optim
include("kinetics_calculator.jl")
include("newton_minimize.jl")

function new_point(param_last, param_index, sign; threshold=0.025, q=0.1)
    stop_flag = false
    step_size = zeros(length(param_last))
    step_size[param_index] = 10 # behöver ändras
    for i in 1:50
        if abs(f(param_last + sign .* step_size) - f(param_last) - q * threshold) < 1e-3
            break
        elseif i == 50
            stop_flag = true
        end
        step_size[param_index] /= 2
    end

    new_point = param_last .+ sign .* step_size
    return new_point, stop_flag
end

function pl_costfunction_calc(x_hat, fix_index, fix_value)
    x = insert!(x_hat, fix_index, fix_value)
    pl_calc = cost_function(problem_object, x, experimental_data)

    return pl_calc
end

function intermediate_cost_function(x_small, index_x_small, x_big)
    x_big_ = convert.(eltype(x_small), x_big)
    x_big_[index_x_small] .= x_small
    return cost_function(problem_object, x_big_, experimental_data)
end

# Define the function to perform profile likelihood analysis for one parameter
function profile_likelihood(params, data, param_index, bounds, num_points, threshold)

    # Save the optimized values for the other three parameters
    other_params = [params[i] for i in 1:4 if i != param_index]

    # list of indexes to be optimized
    index_list = [i for i in 1:4 if i != param_index]

    # new bounds
    new_bounds = deleteat!(bounds, param_index)

    # new start values
    x_samples = readdlm("latin_hypercube.csv", Float64)
    new_x_samples = hcat(x_samples[:, 1:param_index-1], x_samples[:, param_index+1:end])

    stop_flag = false
    sign = -1
    fix_param_list = zeros(num_points)
    x_list = zeros(num_points) #define union typ
    costfunc_value_list = zeros(num_points)

    i = 0

    while i <= num_points
        i += 1

        if stop_flag == false
            # calculate next point
            params_current, stop_flag = new_point(params, i, sign)

        elseif stop_flag == true && sign == -1
            sign = 1
            i = 0
        else
            break
        end

        # Omdefinera kostfuntionen
        cost_function_profilelikelihood = (x) -> intermediate_cost_function(x, index_list, params_current)

        # Find the maximum likelihood estimate for the parameter of interest
        x_min, f_min = p_est(cost_function_profilelikelihood, new_bounds, 100)

        if sign == -1
            fix_param_list[Int(num_points / 2)+1-i] = params_current[param_index]
            x_list[Int(num_points / 2)+1-i] = params_current
            costfunc_value_list[Int(num_points / 2)+1-i] = cost_function_profilelikelihood(params_current)
        else
            fix_param_list[Int(num_points / 2)+i] = params_current[param_index]
            x_list[Int(num_points / 2)+i] = params_current
            costfunc_value_list[Int(num_points / 2)+i] = cost_function_profilelikelihood(params_current)
        end
    end

    # sätta in data i struct, ta bort nollor samtidigt
    return #nått kul
end

function run_profile_likelihood(params, data, bounds, num_points, threshold)
    for i in 1:length(bounds)
        result = profile_likelihood(params, data, i, bounds, num_points, threshold)
    end
end

# Define the initial parameter values
params = [0.9 0.53 3.05 9.93]

# Perform profile likelihood analysis for each parameter
num_points = 100
threshold = 1.92 # For 95% confidence interval
