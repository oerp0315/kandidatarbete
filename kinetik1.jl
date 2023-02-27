using DifferentialEquations, ModelingToolkit

#Funktion för att köra modellen givet parametrar och initialvärden
function kineticmodell1simulator(kgissning,cvidt0)
      @parameters t k[1:4]     #Parametrar i modellen
      @variables c1(t) c2(t) c3(t)   #Variabler i modellen
      D = Differential(t) #Definierar tecken för derivata

      eqs = [D(c1) ~ -k[1]*c1+k[2]*c2,
            D(c2) ~ k[1]*c1-k[2]*c2-k[3]*c2+k[4]*c3,
            D(c3) ~ k[3]*c2-k[4]*c3]  #Uttryck för systemet som diffrentialekvationer


      @named sys = ODESystem(eqs) #Definierar av som är systemet från diffrentialekvationerna
      sys = structural_simplify(sys) #Skriver om systemet så det blir lösbart

      u0 = [c1=>cvidt0[1],
            c2=>cvidt0[2],
            c3=>cvidt0[3]] #Definierar initialvärden

      p = [k[1] =>kgissning[1],
           k[2] =>kgissning[2],
           k[3] =>kgissning[3],
           k[4] =>kgissning[4]] #Definierar värden för parametrarna

      tspan = (0.0, 30.0) #Tiden vi kör modellen under
      prob = ODEProblem(sys, u0, tspan, p, jac = true)  #Definierar vad som ska beräknas
      sol = solve(prob)  #Beräknar lösningen
      return sol
end


kgissning = [1 0.5 3 10] # Gissar parametervärden
cvidt0 = [0.5 0 0.5] #Intialkoncentrationer
sol = kineticmodell1simulator(kgissning,cvidt0) #Kör modellen


# Kör experiment
using Random, Distributions
function experimenter(ttest,numberofexperiments)
      kgissning = [1 0.5 3 10]
      cvidt0 = [0.5 0 0.5]
      sol = kineticmodell1simulator(kgissning,cvidt0)
      
      cslut=zeros(Float64,3,numberofexperiments)
      tslut=zeros(Float64,1,numberofexperiments)

      # Introducerar brus i datan
      for n =1:numberofexperiments
            noisedistribution = Normal(0,0.03)
            i = findfirst(t-> t>=ttest, sol.t)
            cslut[:,n]= sol[:,i]+rand(noisedistribution,3)
            tslut[n] = sol.t[i]
      end

      return cslut,tslut

end

cslut,tslut =experimenter(10,2)

using Plots
plot(sol, idxs = (t, [c1 c2 c3])) #Plottar lösningen
plot!([tslut tslut tslut]',cslut,seriestype=:scatter) #Plottar experimenten


struct experimentresults
      c::Matrix{Float64}
      a_float::Vector{Float64}
end

function minimeringsfunktion(k)
      cvidt0 = [0.5 0 0.5] #Intialkoncentrationer
      sol= kineticmodell1simulator(k,cvidt0)

end

