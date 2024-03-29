using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions
include("newton_minimize.jl")
include("profile_likelihood.jl")

# Object for experimental results
struct experiment_results
    c0::AbstractVector
    c_final::AbstractVector
    t_final::Number
end

"Model construction"
function model_2p_initialize()
    @parameters t θ[1:2]     #Parametrar i modellen
    @variables c1(t) c2(t)   #Variabler i modellen
    D = Differential(t) #Definierar tecken för derivata

    equation_system = [D(c1) ~ -θ[1] * c1 + θ[2] * c2,
        D(c2) ~ θ[1] * c1 - θ[2] * c2]  #Uttryck för systemet som differentialekvationer

    @named system = ODESystem(equation_system) #Definierar av som är systemet från diffrentialekvationerna
    system = structural_simplify(system) #Skriver om systemet så det blir lösbart

    # Intialvärden som kommer skrivas över
    c0 = [0, 0]
    θin = [0, 0]

    u0 = [c1 => c0[1],
        c2 => c0[2]] #Definierar initialvärden

    p = [θ[1] => θin[1],
        θ[2] => θin[2]] #Definierar värden för parametrarna

    tspan = (0.0, 10) #Tiden vi kör modellen under
    problem_object = ODEProblem(system, u0, tspan, p, jac=true)  #Definierar vad som ska beräknas
    return problem_object
end

"Solve the ODE system"
function model_solver(_problem_object, θin, c0, t_stop)
    problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_stop), p=θin)
    solution = solve(problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8, maxiters=1e5)
    return solution
end

"Catch error"
function check_error(e)
    if e isa BoundsError
        @warn "Bounds error ODE solve"
    elseif e isa DomainError
        @warn "Domain error on ODE solve"
    elseif e isa SingularException
        @warn "Singular exeption on ODE solve"
    else
        rethrow(e)
    end
end

"Calculate difference between experiments and model"
function cost_function(problem_object, logθ, experimental_data::AbstractVector)
    θ = exp.(logθ)
    error = 0
    for data in experimental_data
        success = true
        try
            sol = model_solver(problem_object, θ, data.c0, data.t_final)
            if !(sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated)
                success = false
            end
            if success
                c_final_model = sol.u[end]
                error += sum((c_final_model - data.c_final) .^ 2)
            end
        catch e
            check_error(e)
            success == false
        end
        if success == false
            return Inf
        end
    end
    return error
end


"Run experiment"
function experimenter(problem_object, t_stop, c0, standard_deviation, θin)
    solution = model_solver(problem_object, θin, c0, t_stop) #Genererar lösningar
    noise_distribution = Normal(0, standard_deviation) #Skapar error
    return solution[:, end] + rand(noise_distribution, length(c0)) # Lägger till error
end

"Generate experimental data"
function random_dataset_generator(problem_object, number_of_experiments, θin; standard_deviation=0.01)
    experimental_data = []
    for i = 1:number_of_experiments
        Random.seed!(10 * i)
        t_final_data = rand() #Genererar slumpmässiga sluttider
        c0_data = rand!(zeros(length(problem_object.u0))) #Genererar slumpmässiga intial koncentrationer
        c_final_data = experimenter(problem_object, t_final_data, c0_data, standard_deviation, θin)
        current_data = experiment_results(c0_data, c_final_data, t_final_data)
        push!(experimental_data, current_data)
    end
    return experimental_data
end

"Plot true solution"
function plot_exact_example(problem_object, θin)
    c0 = [0.5, 0.0] # Initial concentrations
    sol = model_solver(problem_object, θin, c0, 1) #Kör modellen
    plot(sol, xaxis="Tid (s)", yaxis="Koncentration", label=[L"c_{A}" L"c_{B}"], lw=2, legendfontsize=15, labelfontsize=15) #Plottar lösningen
    savefig("exact_ex.png")
end

"Plot model"
function plot_experiment(experimental_data)
    for data in experimental_data
        plot!(data.t_final * ones(length(data.c_final)), data.c_final, seriestype=:scatter) #Plottar lösningen
    end
    savefig("exp_data.png")
end

problem_object = model_2p_initialize()
experimental_data = random_dataset_generator(problem_object, 50, [3.0, 3.0])

bounds = [(0.01, 6), (0.01, 6)]
log_bounds = map(x -> (log(x[1]), log(x[2])), bounds)

f(x) = cost_function(problem_object, x, experimental_data)

# run the parameter estimation
x_min, f_min = p_est(f, log_bounds, 10, false)

plot_exact_example(problem_object, [3.0, 3.0])
plot_experiment(experimental_data)

# Define the initial parameter values
params = x_min

# Perform profile likelihood analysis for each parameter
num_points = 100
threshold = 3.84

# save threshold
CSV.write("profilelikelihood_results/threshold.csv", DataFrame(threshold=threshold))

run_profile_likelihood(params, 1000, bounds, num_points, threshold)

contourplot_2p()
