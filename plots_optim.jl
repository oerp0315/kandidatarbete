using Plots
using CSV
using DataFrames

function plot_pl()
    data = CSV.read("profilelikelihood_results/profile_likelihood.csv", DataFrame;
        header=[:fixed_parameter_index, :fixed_parameter, :parameters, :cost_function_values])

    grouped = groupby(data, :fixed_parameter_index)
    # Iterate over groups and create a plot for each one
    for (sample_number, group) in pairs(grouped)
        sample_number = sample_number.fixed_parameter_index

        # plot profile likelihood
        plot(group[!, :fixed_parameter], group[!, :cost_function_values], xaxis="Parameter $sample_number", yaxis="Cost function values", legend=false, lc=:black, lw=2)

        # plot threshold
        x = collect(Float64, range(group[!, :fixed_parameter][1], group[!, :fixed_parameter][end], length=2))
        y = (CSV.read("p_est_results/opt_point.csv", DataFrame)[1, 2] + CSV.read("profilelikelihood_results/threshold.csv", DataFrame)[1, 1]) * ones(length(x))
        plot!(x, y, lc=:black, linestyle=:dash, lw=2)

        #save plot
        savefig("profilelikelihood_results/parameter$sample_number.png")
    end
end

function plot_waterfall()
    data = CSV.read("p_est_results/waterfall_data.csv", DataFrame, types=Float64)
    sort!(data)
    y = data[:, 1]
    x = collect(1:length(y))

    plot(x, y)
    savefig("p_est_results/waterfall_plot")
end

# plot profile likelihood
plot_pl()

# plot waterfall plot
plot_waterfall()
