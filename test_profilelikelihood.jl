using Optim
include("kinetics_calculator.jl")
include("newton_minimize.jl")

function step_size(param_last, param_index; threshold=0.025, q=0.1)
    step_size = zeros(length(param_last))
    step_size[param_index] = 10
    for i in 1:50
        if abs(f(param_last + step_size) - f(param_last) - q * threshold) < 1e-3
            break
        else
            step_size[param_index] /= 2
        end
    end

    return param_last .+ step_size
end

function pl_costfunction_calc(x_hat, fix_index, fix_value)
    x = insert!(x_hat, fix_index, fix_value)
    pl_calc = cost_function(problem_object, x, experimental_data)

    return pl_calc
end

function intermediate_cost_function(x_small, index_x_small, x_big)
    _x_big = convert.(eltype(x_small), x_big)
    _x_big[index_x_small] .= x_small
    return cost_function(problem_object, _x_big, experimental_data)
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
    new_x_samples = hcat(bounds[:, 1:param_index-1], A[:, param_index+1:end])

    for i in 1:100
        # Omdefinera när det behövs
        cost_function_profilelikelihood = (x) -> intermediate_cost_function(x, index_list, params)

        params_current = step_size(params, param_index)

        # Find the maximum likelihood estimate for the parameter of interest
        x_min, f_min = p_est(cost_function_profilelikelihood, new_bounds, 100)
    end

    # Find the maximum likelihood estimates for the other three parameters at each value of the parameter of interest
    profile_likelihood_values = []
    for i in 1:num_points
        param_value = linspace(params[param_index] - threshold, params[param_index] + threshold, num_points)[i]
        other_param_values = []
        for j in 1:3
            result = optimize(objective, [other_params[j]], method=GoldenSection())
            push!(other_param_values, result.minimizer[1])
        end
        likelihood_value = likelihood([other_param_values; param_value], data)
        push!(profile_likelihood_values, likelihood_value)
    end

    # Find the confidence interval for the parameter of interest
    max_likelihood = maximum(profile_likelihood_values)
    ci_lower = params[param_index] - threshold + (findfirst(profile_likelihood_values .>= max_likelihood - threshold) - 1) * threshold / num_points
    ci_upper = params[param_index] - threshold + findlast(profile_likelihood_values .>= max_likelihood - threshold) * threshold / num_points

    # Return the results
    return mle, ci_lower, ci_upper, profile_likelihood_values
end

# Define the initial parameter values
params = [0.9 0.53 3.05 9.93]

# Perform profile likelihood analysis for each parameter
num_points = 100
threshold = 1.92 # For 95% confidence interval
results = []
for i in 1:4
    mle, ci_lower, ci_upper, profile_likelihood_values = profile_likelihood(params, experimental_data, i, num_points, threshold)
    push!(results, (mle, ci_lower, ci_upper, profile_likelihood_values))
end

# Print the results
for i in 1:4
    println("Parameter $i:")
    println("MLE: $(results[i][1])")
    println("CI: ($(results[i][2]), $(results[i][3]))")
    println("Profile likelihood values: $(results[i][4])")
end
