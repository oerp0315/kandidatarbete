using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions

# Skriv om
"Object for experimental results"
struct experiment_results
      c_final::AbstractVector
      t_final::Number
end

#=
"Function to construct model"
function model_initialize()
    @parameters t Extracellular_glucose k_{t,Snf3} k_{t,Snf3} k_{a,Snf3} k_{i,Snf3g} k_{t,Std1} k_{d,Std1}  k_{a,Std1:Rgt1} k_{i,Std1:Rgt1} k_{t,Mth1} k_{d,Mth1} k_{a,Mth1:Rgt1} k_{i,Mth1:Rgt1}  V_{mSNF3} θ_{Mig1-Snf3} θ_{Mig2-Snf3} V_{mSTD1} θ_{Rgt1_active-Snf3} V_{mMTH1} θ_{Rgt1_active-MTH1} θ_{Mig1-MTH1} θ_{Mig2-MTH1} V_{mRGT1} k_{t,Hxt1} k_{t,Hxt2} k_{t,Hxt3} k_{t,Hxt4} k_{d,Hxt1} k_{d,Hxt2} k_{d,Hxt3} k_{d,Hxt4} k_{a,ADP} k_{d,ADP} k_{a,ATP} k_{t,Snf1} k_{d,Snf1} k_{i,Snf1} k_{t,Mig1} k_{d,Mig1} k_{i,Mig1} k_{t,Mig2} k_{d,Mig2} V_{transport-Hxt1} K_{transport-Hxt1} V_{transport-Hxt2} K_{transport-Hxt2} V_{transport-Hxt3} K_{transport-Hxt3} V_{transport-Hxt4} K_{transport-Hxt4} k_{p,ATP} k_{d,mHXT1} V_{mHXT1} T_{mHXT1} θ_{activation} θ_{Rgt1_active-HXT1} k_{d,mHXT2} V_{mHXT2} θ_{Rgt1_active-HXT2} θ_{Mig1-HXT2} θ_{Mig2-HXT2} k_{d,mHXT3} V_{mHXT3} θ_{Rgt1_active-HXT3} θ_{Mig1-HXT3} θ_{Mig2-HXT3} k_{d,mHXT4} V_{mHXT4} θ_{Rgt1_active-HXT4} θ_{Mig1-HXT4} θ_{Mig2-HXT4} k_{d,mMIG1} V_{mMIG1} θ_{Mig1-MIG1} θ_{Mig2-MIG1} k_{d,mMIG2} V_{mMIG2} θ_{Rgt1_active-MIG2} θ_{Mig1-MIG2} θ_{Mig2-MIG2} k_{d,mSNF1} V_{mSNF1}
    @variables  Snf3(t) Snf3g(t) Std1(t) Std1:Rgt1(t) Mth1(t) Mth1:Rgt1(t) Mth1:Rgt1(t)  Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) mHXT1(t) Hxt1(t) mHXT2(t) Hxt2(t) mHXT3(t) Hxt3(t) mHXT4(t) Hxt4(t) ATP(t) ADP(t) mSNF1(t) Snf1(t) Cellular_glucose(t) mMIG1(t) Mig1(t) mMIG2(t) Mig2(t) Rgt1(t) #Variabler i modellen
    D = Differential(t)

    #gener 
    equation_system = [D(Snf3) ~ k_{t,Snf3}*mSNF3- k_{d,Snf3}*Snf3 - k_{a,Snf3}*Snf3*Extracellular_glucose + k_{i,Snf3g}*Snf3g,
        D(Snf3g) ~ k_{a,Snf3}*Snf3*Extracellular_glucose - k_{i,Snf3}*Snf3g,
        D(Std1) ~ k_{t,Std1}*mSTD1 - k_{d,Std1}*Std1- k_{a,Std1:Rgt1}*std1*Rgt1+ k_{i,Std1:Rgt1}*Std1:Rgt1,
        D(Mth1) ~ k_{t,Mth1}*mMTH1 - k_{d,Mth1}*Mth*Snf3g- k_{a,Mth1:Rgt1}*Rgt1*Mth1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1,
        D(Rgt1) ~ k_{t,Rgt1}*mRGT1 - k_{d,Rgt1}*Rgt1 - k_{a,Std1:Rgt1}*Std1*Rgt1 + k_{i,Std1:Rgt1}*Std1:Rgt1 - k_{a,Mth1:Rgt1}*Mth1*Rgt1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1, # Annat utryck om vi testar fosforylering
        D(Std1:Rgt1) ~ k_{a,Std1:Rgt1} *Std1*Rgt1 - k_{i,Std1:Rgt1}*Std1:Rgt1,
        D(Mth1:Rgt1) ~ k_{a,Mth1:Rgt1} *Mth1*Rgt1 - k_{i,Mth1:Rgt1}*Mth1:Rgt1,

        D(Hxt1)~k_{t,Hxt1}*mHXT1 - k_{d,Hxt1}*Hxt1, 
        D(Hxt2)~k_{t,Hxt2}*mHXT2 - k_{d,Hxt2}*Hxt2, 
        D(Hxt3)~k_{t,Hxt3}*mHXT3 - k_{d,Hxt3}*Hxt3,
        D(Hxt4)~k_{t,Hxt4}*mHXT4 - k_{d,Hxt4}*Hxt4, 
        D(Hxt5)~k_{t,Hxt5}*mHXT5 - k_{d,Hxt5}*Hxt5, 
        D(Hxt6)~k_{t,Hxt6}*mHXT6 - k_{d,Hxt6}*Hxt6, 
        D(ADP)~ k_{a,ADP}*ATP - k_{d,ADP}*ADP - k_{a,ATP}*ADP*Extracellular_glucose,
        D(Snf1)~ k_{t,Snf1}*mSNF1- k_{d,Snf1}*Snf1+ k_{i,Snf1}*Snf1*Cellular_glucose,
        D(Mig1)~ k_{t,Mig1}*mMIG1 - k_{d,Mig1}*Mig1 -  k_{i,Mig1}*Mig1*Snf1,
        D(Mig2)~k_{t,Mig2}*mMIG2 - k_{d,Mig2}*Mig2,
        D(Cellular_glucose)~ V_{transport-Hxt1}*Extracellular_glucose/(K_{transport-Hxt1} + Extracellular_glucose) + V_{transport-Hxt2}*Extracellular_glucose/(K_{transport-Hxt2} + Extracellular_glucose) + V_{transport-Hxt3}*Extracellular_glucose/(K_{transport-Hxt3} + Extracellular_glucose) + V_{transport-Hxt4}*Extracellular_glucose/(K_{transport-Hxt4} + Extracellular_glucose) - k_{p,ATP}*Cellular_glucose, 

        Rgt1_active ~ Std1:Rgt1 + Mth1:Rgt1,

        #proteiner mRNA
        D(mSNF3) ~ -k_{d,mSNF3}*mSNF3 + V_{mSNF3}/(1+θ_{Mig1-Snf3}*Mig1)/(1+θ_{Mig2-Snf3}*Mig2),
        D(mSTD1) ~ -k_{d,mSTD1}*mSTD1 + V_{mSTD1}/(1+θ_{Rgt1_active-Snf3}*Rgt1_active),
        D(mMTH1) ~ -k_{d,mMTH1}*mMTH1 + V_{mMTH1}/(1+θ_{Rgt1_active-MTH1}*Rgt1_active)/(1+θ_{Mig1-MTH1}*Mig1)/(1+θ_{Mig2-MTH1}*Mig2),
        D(mRGT1) ~ - k_{d,mRGT1}*mRGT1 + V_{mRGT1},

        D(mHXT1) ~ -k_{d,mHXT1}*mHXT1 + V_{mHXT1}*(T_{mHXT1}+((1-T_{mHXT1})*θ_{activation}*Rgt1*Glucosesignal)/(1+θ_{activation}*Rgt1*Glucosesignal))/(1+θ_{Rgt1_active-HXT1}*Rgt1_active), #läs på om glucosesignal - för ska ej vara med, kolla på systemet och se vilka antagande som kan göras, hitta därefter parameter som kan ersätta
        D(mHXT2) ~ - k_{d,mHXT2}*mHXT2 + V_{mHXT2}/(1+θ_{Rgt1_active-HXT2}*Rgt1_active)/(1+θ_{Mig1-HXT2}*Mig1)/(1+θ_{Mig2-HXT2}*Mig2),
        D(mHXT3) ~ -k_{d,mHXT3}*mHXT3 + V_{mHXT3}/(1+θ_{Rgt1_active-HXT3}*Rgt1_active)/(1+θ_{Mig1-HXT3}*Mig1)/(1+θ_{Mig2-HXT3}*Mig2),
        D(mHXT4) ~ -k_{d,mHXT4}*mHXT4 + V_{mHXT4}/(1+θ_{Rgt1_active-HXT4}*Rgt1_active)/(1+θ_{Mig1-HXT4}*Mig1)/(1+θ_{Mig2-HXT4}*Mig2),
        D(mHXT5) ~ -k_{d,mHXT5}*mHXT5 + V_{mHXT5}/(1+θ_{Rgt1_active-HXT5}*Rgt1_active)/(1+θ_{Mig1-HXT5}*Mig1)/(1+θ_{Mig2-HXT5}*Mig2),
        D(mHXT6) ~ -k_{d,mHXT6}*mHXT6 + V_{mHXT3}/(1+θ_{Rgt1_active-HXT6}*Rgt1_active)/(1+θ_{Mig1-HXT6}*Mig1)/(1+θ_{Mig2-HXT6}*Mig2),
        D(mMIG1) ~ -k_{d,mMIG1}*mMIG1 + V_{mMIG1}/(1+θ_{Mig1-MIG1}*Mig1)/(1+θ_{Mig2-MIG1}*Mig2),
        D(mMIG2) ~ -k_{d,mMIG2}*mMIG2 + V_{mMIG2}/(1+θ_{Rgt1_active-MIG2}*Rgt1_active)/(1+θ_{Mig1-MIG2}*Mig1)/(1+θ_{Mig2-MIG2}*Mig2),
        D(mSNF1) ~ -k_{d,mSNF1}*mSNF1 + V_{mSNF1}]

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
=# 

# Skriv om!!
"Calculate difference between experiments and model"
function cost_function(problem_object, logθ, experimental_data::AbstractVector)
      θ = exp.(logθ)
      error = 0
      for data in experimental_data
            sol = model_solver1(problem_object, θ, data.c0, data.t_final)

            c_final_model = sol.u[end]
            error += sum((c_final_model - data.c_final) .^ 2)
      end
      return error
end

function interpolate(t, t_1, t_2, f_1, f_2)
    return f_1 + (t-t_1) * (f_1 -f_2)/(t_1-t_2)
end
