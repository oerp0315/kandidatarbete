using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions

# Objekt för experimentresultat
struct results
      c0::AbstractVector
      cslut::AbstractVector
      ts::Number
end

#Funktion för att köra modellen givet parametrar och initialvärden
function modell1simulator(θin,c0,ts)
      @parameters t θ[1:4]     #Parametrar i modellen
      @variables c1(t) c2(t) c3(t)   #Variabler i modellen
      D = Differential(t) #Definierar tecken för derivata

      eqs = [D(c1) ~ -θ[1]*c1+θ[2]*c2,
            D(c2) ~ θ[1]*c1-θ[2]*c2-θ[3]*c2+θ[4]*c3,
            D(c3) ~ θ[3]*c2-θ[4]*c3]  #Uttryck för systemet som diffrentialekvationer


      @named sys = ODESystem(eqs) #Definierar av som är systemet från diffrentialekvationerna
      sys = structural_simplify(sys) #Skriver om systemet så det blir lösbart

      u0 = [c1=>c0[1],
            c2=>c0[2],
            c3=>c0[3]] #Definierar initialvärden

      p = [θ[1] =>θin[1],
           θ[2] =>θin[2],
           θ[3] =>θin[3],
           θ[4] =>θin[4]] #Definierar värden för parametrarna

      tspan = (0.0, ts) #Tiden vi kör modellen under
      prob = ODEProblem(sys, u0, tspan, p, jac = true)  #Definierar vad som ska beräknas
      sol = solve(prob)  #Beräknar lösningen
      return sol
end

# Kör experiment
function experimenter(ts,c0; θin = [1 0.5 3 10], seed=1 )
      Random.seed!(seed)
      sol = modell1simulator(θin,c0,ts)
      
      noisedistribution = Normal(0,0.03)
      cslut = sol[:,end] + rand(noisedistribution,3)

      return cslut
end


function kostnadsfunktion(θ,Data::AbstractVector)
      error=0
      for data in Data
            sol = modell1simulator(θ,data.c0,data.ts)
            cs = sol.u[end]
            error +=sum((cs-data.cslut).^2)
      end
      print(typeof(error))
      return error
end


θin = [1,0.5,3,10] # Gissar parametervärden
c0 = [0.5,0,0.5] #Intialkoncentrationer
sol = modell1simulator(θin,c0,30) #Kör modellen
plot(sol) #Plottar lösningen


# Genererar exprimentresultat
tslut = 10
cresultat1 = experimenter(tslut,c0)
plot!([tslut tslut tslut]',cresultat1,seriestype=:scatter) #Plottar experimenten

# Packeterar data till experimentresultat
data1 = results(c0,cresultat1,tslut)
Data = [data1]
println(kostnadsfunktion([2,0.5,3,10],Data))
