using Plots
using CSV
using DataFrames
using LaTeXStrings



function plot_pl()
    data = CSV.read("profilelikelihood_results/profile_likelihood.csv", DataFrame;
        header=[:fixed_parameter_index, :fixed_parameter, :parameters, :cost_function_values])

    grouped = groupby(data, :fixed_parameter_index)
    # Iterate over groups and create a plot for each one
    for (sample_number, group) in pairs(grouped)
        sample_number = sample_number.fixed_parameter_index

        # plot profile likelihood
        plot(group[!, :fixed_parameter], group[!, :cost_function_values], xaxis=L"$\theta_%$sample_number$", yaxis="Kostnadsfunktion", legend=false, lc=:black, lw=2, labelfontsize=15)

        # plot threshold
        x = collect(Float64, range(group[!, :fixed_parameter][1], group[!, :fixed_parameter][end], length=2))
        y = (CSV.read("p_est_results/opt_point.csv", DataFrame)[1, 2] + CSV.read("profilelikelihood_results/threshold.csv", DataFrame)[1, 1]) * ones(length(x))
        plot!(x, y, lc=:black, linestyle=:dash, lw=2)

        #save plot
        savefig("profilelikelihood_results/parameter$sample_number.png")
    end
end

function plot_waterfall()
    data = CSV.read("p_est_results/sample_data.csv", DataFrame)
    y = data[:, 5]
    sort!(y)
    x = collect(1:length(y))
    #y = log.(y)


    plot(x[1:end-300], y[1:end-300], xaxis="Startvärden", yaxis="Kostnadsfunktion", lw=2, labelfontsize=15, legend=false,  linewidth = 4, c=RGB(0.41, 0.82, 0.91))
    savefig("p_est_results/waterfall_plot")

    n_convergent_samples = 0
    for i in 1:length(x)
        if abs(y[i] - y[1]) < 0.1
            n_convergent_samples += 1
        else
            break
        end
    end
    convergence_ratio = 100 * (n_convergent_samples / length(x))
    result = "$convergence_ratio% of the start points converged to the same minimum point, 
with an accepted difference of 0.1 in function value from the smallest function value"
    CSV.write("p_est_results/waterfall_results.csv", DataFrame(waterfall_result=result))
end

# plot profile likelihood
plot_pl()

# plot waterfall plot
plot_waterfall()
