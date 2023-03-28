using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions




# Funktion för att lösa ODE-problem med givna parametrar, intialvärden och sluttid
function model_solver(_problem_object, θin, c0, t_stop)
      problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_stop), p=θin)
      solution = solve(problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8)
      return solution
end






#plot()
#plot_exact_example(problem_object)
#plot_experiment(experimental_data)
#display(plt)

# printar kostnaden av exakta punkten. Borde ge 0.0
# println(cost_function(problem_object, [1, 0.5, 3, 10], experimental_data))

