using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions

#Random.seed!(12)

println("Nu kör vi!!")

# Objekt för experimentresultat
struct experiment_results
      c0::AbstractVector
      c_final::AbstractVector
      t_final::Number
end

#Funktion för att köra modellen givet parametrar och initialvärden
function modellsimulator1(θin,c0,t_stop)
      @parameters t θ[1:4]     #Parametrar i modellen
      @variables c1(t) c2(t) c3(t)   #Variabler i modellen
      D = Differential(t) #Definierar tecken för derivata

      equation_system = [D(c1) ~ -θ[1]*c1+θ[2]*c2,
            D(c2) ~ θ[1]*c1-θ[2]*c2-θ[3]*c2+θ[4]*c3,
            D(c3) ~ θ[3]*c2-θ[4]*c3]  #Uttryck för systemet som diffrentialekvationer


      @named system = ODESystem(equation_system) #Definierar av som är systemet från diffrentialekvationerna
      system = structural_simplify(system) #Skriver om systemet så det blir lösbart

      u0 = [c1=>c0[1],
            c2=>c0[2],
            c3=>c0[3]] #Definierar initialvärden

      p = [θ[1] =>θin[1],
           θ[2] =>θin[2],
           θ[3] =>θin[3],
           θ[4] =>θin[4]] #Definierar värden för parametrarna

      tspan = (0.0, t_stop) #Tiden vi kör modellen under
      prob = ODEProblem(system, u0, tspan, p, jac = true)  #Definierar vad som ska beräknas
      sol = solve(prob,Rodas5())  #Beräknar lösningen
      return sol
end

# Kör experiment
function experimenter(t_final,c0; θin = [1 0.5 3 10], standarddeviation=0)
      sol = modellsimulator1(θin,c0,t_final) #Genererar lösningar
      noise_distribution = Normal(0,standarddeviation) #Skapar error
      return sol[:,end] + rand(noise_distribution,3) # Lägger till error
end


function kostnadsfunktion(θ,experimental_data::AbstractVector)
      error=0
      for data in experimental_data
            sol = modellsimulator1(θ,data.c0,data.t_final)
            c_final_model = sol.u[end]
            error +=sum((c_final_model-data.c_final).^2)
      end
      return error
end


experimental_data = []
for i = 1:2
      t_final_data = 2*rand()
      c0_data = [rand(),rand(),rand()]
      c_final_data=experimenter(t_final_data, c0_data)

      current_data = experiment_results(c0_data,c_final_data, t_final_data)
      push!(experimental_data, current_data)
end



println(kostnadsfunktion([1,0.5,3,10],experimental_data))



# För att Plotta

θin = [1,0.5,3,10] # Gissar parametervärden
c0 = [0.5,0,0.5] #Intialkoncentrationer
sol = modellsimulator1(θin,c0,2) #Kör modellen
plot(sol) #Plottar lösningen


# Genererar exprimentresultat
#tslut = 10
#cresultat1 = experimenter(tslut,c0)
#plot!([tslut tslut tslut]',cresultat1,seriestype=:scatter) #Plottar experimenten

# Packeterar data till experimentresultat
#data1 = results(c0,cresultat1,tslut)