using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions

println("Nu kör vi!!")

# Objekt för experimentresultat
struct experiment_results
      c0::AbstractVector
      c_final::AbstractVector
      t_final::Number
end


# Funktion för att initiera ODE modell
function model_initialize1()
      @parameters t θ[1:4]     #Parametrar i modellen
      @variables c1(t) c2(t) c3(t)   #Variabler i modellen
      D = Differential(t) #Definierar tecken för derivata

      equation_system = [D(c1) ~ -θ[1] * c1 + θ[2] * c2,
            D(c2) ~ θ[1] * c1 - θ[2] * c2 - θ[3] * c2 + θ[4] * c3,
            D(c3) ~ θ[3] * c2 - θ[4] * c3]  #Uttryck för systemet som diffrentialekvationer


      @named system = ODESystem(equation_system) #Definierar av som är systemet från diffrentialekvationerna
      system = structural_simplify(system) #Skriver om systemet så det blir lösbart

      # Intialvärden som kommer skrivas över
      c0 = [0, 0, 0]
      θin = [0, 0, 0, 0]

      u0 = [c1 => c0[1],
            c2 => c0[2],
            c3 => c0[3]] #Definierar initialvärden

      p = [θ[1] => θin[1],
            θ[2] => θin[2],
            θ[3] => θin[3],
            θ[4] => θin[4]] #Definierar värden för parametrarna

      tspan = (0.0, 10) #Tiden vi kör modellen under
      problem_object = ODEProblem(system, u0, tspan, p, jac=true)  #Definierar vad som ska beräknas
      return problem_object
end

# Funktion för att lösa ODE-problem med givna parametrar, intialvärden och sluttid
function model_solver1(_problem_object, θin, c0, t_stop)
      problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_stop), p=θin)
      solution = solve(problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8)
      return solution
end

# Kör experiment
function experimenter(problem_object, t_stop, c0, standard_deviation, θin)
      solution = model_solver1(problem_object, θin, c0, t_stop) #Genererar lösningar
      noise_distribution = Normal(0, standard_deviation) #Skapar error
      return solution[:, end] + rand(noise_distribution, 3) # Lägger till error
end

function cost_function(problem_object, θ, experimental_data::AbstractVector)
      error = 0
      for data in experimental_data
            sol = model_solver1(problem_object, θ, data.c0, data.t_final)
            #sol = modellsimulator1(θ,data.c0,data.t_final)

            c_final_model = sol.u[end]
            error += sum((c_final_model - data.c_final) .^ 2)
      end
      return error
end

function random_dataset_generator(problem_object, number_of_experiments; standard_deviation=0.03, θin=[1 0.5 3 10])
      experimental_data = []
      for i = 1:number_of_experiments
            t_final_data = 2 * rand() #Genererar slumpmässiga sluttider
            c0_data = [rand(), rand(), rand()]    #Genererar slumpmässiga intial koncentrationer
            c_final_data = experimenter(problem_object, t_final_data, c0_data, standard_deviation, θin)

            current_data = experiment_results(c0_data, c_final_data, t_final_data)
            push!(experimental_data, current_data)
      end
      return (experimental_data)
end

function plot_exact_example(problem_object)
      θin = [1, 0.5, 3, 10] # Gissar parametervärden
      c0 = [0.5, 0, 0.5] #Intialkoncentrationer
      sol = model_solver1(problem_object, θin, c0, 2) #Kör modellen
      plot!(sol) #Plottar lösningen
end

function plot_experiment(experimental_data)
      for data in experimental_data
            plot!(data.t_final * ones(length(data.c_final)), data.c_final, seriestype=:scatter) #Plottar lösningen
      end
end

problem_object = model_initialize1()
problem_object = model_initialize1()
experimental_data = random_dataset_generator(problem_object, 2)



plot()
plot_exact_example(problem_object)
#plot_experiment(experimental_data)

for data in experimental_data
      #plot!(data.t_final), data.c_final, seriestype=:scatter) #Plottar lösningen
end

display(plt)

# printar kostnaden av exakta punkten. Borde ge 0.0
println(cost_function(problem_object, [1, 0.5, 3, 10], experimental_data))

