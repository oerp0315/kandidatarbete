using DifferentialEquations, ModelingToolkit, Plots, Random, Distributions

# Skriv om
"Object for experimental results"
struct experiment_results
      glucose_conc ::Number
      hxt_types::AbstractVector
      c:: AbstractMatrix
      t:: AbstractVector
end

index_general = [1,1,2,2,3,3,4,4]
index_mutant = [2,2]

Data01_glucose=[ 0.74  0.1   0.06  0.05  0.76  0.13  23.02  26.98
                1.83  0.52  0.1   0.06  0.85  0.33  29.55  36.75
                0.23  0.02  0.29  0.02  0.95  0.05  41.21  53.44
                0.08  0.05  0.19  0.29  0.24  0.14  31.34  44.63
                0.05  0.19  0.15  1.1   0.32  0.18  28.44  25.82
                0.09  0.03  0.06  0.07  0.27  0.09  12.34  16.08
                0.02  0.02  0.01  0.2   0.28  0.04   9.92  13.89]
Data02_glucose=[ 0.07   0.02  0.06  0.05   0.52  0.13  23.02  26.98
                3.92   1.05  0.09  0.09   0.78  0.71  21.19  24.59
                9.86   8.15  0.18  0.18   0.67  0.57  19.69  20.69
                19.08  15.12  0.32  0.32   1.78  1.46  16.35  17.33
                21.03  20.01  0.35  0.35   3.96  2.96  15.57  14.37
                27.03  24.11  0.91  1.18   7.34  8.34  12.55  13.55
                29.03  31.05  1.03  1.08  11.16  9.16  10.37   8.12]
Data01_mutant=[91.09  81.19
                82.14  74.49
                69.08  68.11
                59.57  64.89
                55.71  56.12
                51.68  60.08]
Data02_mutant=[23.83  20.11
                41.84  34.41
                57.87  51.86
                76.11  65.38
                85.12  77.12
                92.12  81.11]

timevalues_general=[0,10,20,30,40,60,120]
timevalues_mutant=[0,10,27,35,60,120]

experiment1 = experiment_results(0.1, index_general , Data01_glucose, timevalues_general)
experiment2 = experiment_results(0.2, index_general, Data02_glucose, timevalues_general)
#experiment3 = experiment_results(0.1, index_mutant,Data01_mutant, timevalues_mutant)
#experiment3 = experiment_results(0.1, index_mutant, Data02_mutant, timevalues_mutant)

experimental_data = [experiment1, experiment2] #Lägg till experiment 3&4 senare
#experimental_data = [experiment1, experiment2,experiment3,experiment4]

"Function to construct model"
function model_initialize()
    @parameters t Extracellular_glucose k_{a,Snf3}  k_{i,Snf3g}  k_{i,Std1} k_{i,Std1}  k_{i,Snf1}  k_{i,Mig1} k_{p,ATP}   k_{d,mHXT1}  T_{mHXT1}  θ_{activation}  k_{d,mHXT2}  k_{d,mHXT3}   k_{d,mHXT4}   k_{d,mMIG1} k_{d,mMIG2} k_{d,mSNF1}
    @variables  Snf3(t) Snf3g(t) Std1(t) Std1:Rgt1(t) Mth1(t) Mth1:Rgt1(t) Mth1:Rgt1(t)  Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) mHXT1(t) Hxt1(t) mHXT2(t) Hxt2(t) mHXT3(t) Hxt3(t) mHXT4(t) Hxt4(t)  mSNF1(t) Snf1(t) Cellular_glucose(t) mMIG1(t) Mig1(t) mMIG2(t) Mig2(t) Rgt1(t) #Variabler i modellen
    D = Differential(t)

    #gener 
    equation_system = [D(Snf3) ~ k_{t,Snf3}*mSNF3- k_{d,Snf3}*Snf3 - k_{a,Snf3}*Snf3*Extracellular_glucose + k_{i,Snf3g}*Snf3g,
        D(Snf3g) ~ k_{a,Snf3}*Snf3*Extracellular_glucose - k_{i,Snf3}*Snf3g,
        D(Std1) ~ k_{t,Std1}*mSTD1 - k_{d,Std1}*Std1 - k_{i,Std1}*Std1*Snf3g- k_{a,Std1:Rgt1}*std1*Rgt1+ k_{i,Std1:Rgt1}*Std1:Rgt1,
        D(Mth1) ~ k_{t,Mth1}*mMTH1 - k_{d,Mth1}*Mth1 - k_{i,Mth1}*Mth*Snf3g- k_{a,Mth1:Rgt1}*Rgt1*Mth1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1,
        D(Rgt1) ~ k_{t,Rgt1}*mRGT1 - k_{d,Rgt1}*Rgt1 - k_{a,Std1:Rgt1}*Std1*Rgt1 + k_{i,Std1:Rgt1}*Std1:Rgt1 - k_{a,Mth1:Rgt1}*Mth1*Rgt1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1, # Annat utryck om vi testar fosforylering
        D(Std1:Rgt1) ~ k_{a,Std1:Rgt1} *Std1*Rgt1 - k_{i,Std1:Rgt1}*Std1:Rgt1,  # Dessa borde skrivas om som jämnviktsekvationer
        D(Mth1:Rgt1) ~ k_{a,Mth1:Rgt1} *Mth1*Rgt1 - k_{i,Mth1:Rgt1}*Mth1:Rgt1, # Dessa borde skrivas om som jämnviktsekvationer
        Rgt1_active ~ Std1:Rgt1 + Mth1:Rgt1,

        D(Hxt1)~k_{t,Hxt1}*mHXT1 - k_{d,Hxt1}*Hxt1, 
        D(Hxt2)~k_{t,Hxt2}*mHXT2 - k_{d,Hxt2}*Hxt2, 
        D(Hxt3)~k_{t,Hxt3}*mHXT3 - k_{d,Hxt3}*Hxt3,
        D(Hxt4)~k_{t,Hxt4}*mHXT4 - k_{d,Hxt4}*Hxt4, 
        D(Snf1)~ k_{t,Snf1}*mSNF1- k_{d,Snf1}*Snf1+ k_{i,Snf1}*Snf1*Cellular_glucose,
        D(Mig1)~ k_{t,Mig1}*mMIG1 - k_{d,Mig1}*Mig1 -  k_{i,Mig1}*Mig1*Snf1,
        D(Mig2)~k_{t,Mig2}*mMIG2 - k_{d,Mig2}*Mig2,
        D(Cellular_glucose)~ V_{transport-Hxt1}*Extracellular_glucose/(K_{transport-Hxt1} + Extracellular_glucose) + V_{transport-Hxt2}*Extracellular_glucose/(K_{transport-Hxt2} + Extracellular_glucose) + V_{transport-Hxt3}*Extracellular_glucose/(K_{transport-Hxt3} + Extracellular_glucose) + V_{transport-Hxt4}*Extracellular_glucose/(K_{transport-Hxt4} + Extracellular_glucose) - k_{p,ATP}*Cellular_glucose, 


        #mRNA
        D(mSNF3) ~ -k_{d,mSNF3}*mSNF3 + V_{mSNF3}/(1+θ_{Mig1-Snf3}*Mig1)/(1+θ_{Mig2-Snf3}*Mig2),
        D(mSTD1) ~ -k_{d,mSTD1}*mSTD1 + V_{mSTD1}/(1+θ_{Rgt1_active-Std1}*Rgt1_active),
        D(mMTH1) ~ -k_{d,mMTH1}*mMTH1 + V_{mMTH1}/(1+θ_{Rgt1_active-MTH1}*Rgt1_active)/(1+θ_{Mig1-MTH1}*Mig1)/(1+θ_{Mig2-MTH1}*Mig2),
        D(mRGT1) ~ - k_{d,mRGT1}*mRGT1 + V_{mRGT1},

        D(mHXT1) ~ -k_{d,mHXT1}*mHXT1 + V_{mHXT1}*(T_{mHXT1}+((1-T_{mHXT1})*θ_{activation}*Rgt1)/(1+θ_{activation}*Rgt1))/(1+θ_{Rgt1_active-HXT1}*Rgt1_active), # Vi har tagit bort glucose signals effekt. Läs på om basalreguleringen
        D(mHXT2) ~ - k_{d,mHXT2}*mHXT2+ V_{mHXT2}/(1+θ_{Rgt1_active-HXT2}*Rgt1_active)/(1+θ_{Mig1-HXT2}*Mig1)/(1+θ_{Mig2-HXT2}*Mig2),
        D(mHXT3) ~ -k_{d,mHXT3}*mHXT3 + V_{mHXT3}/(1+θ_{Rgt1_active-HXT3}*Rgt1_active)/(1+θ_{Mig1-HXT3}*Mig1)/(1+θ_{Mig2-HXT3}*Mig2),
        D(mHXT4) ~ -k_{d,mHXT4}*mHXT4 + V_{mHXT4}/(1+θ_{Rgt1_active-HXT4}*Rgt1_active)/(1+θ_{Mig1-HXT4}*Mig1)/(1+θ_{Mig2-HXT4}*Mig2),
        D(mMIG1) ~ -k_{d,mMIG1}*mMIG1 + V_{mMIG1}/(1+θ_{Mig1-MIG1}*Mig1)/(1+θ_{Mig2-MIG1}*Mig2),
        D(mMIG2) ~ -k_{d,mMIG2}*mMIG2 + V_{mMIG2}/(1+θ_{Rgt1_active-MIG2}*Rgt1_active)/(1+θ_{Mig1-MIG2}*Mig1)/(1+θ_{Mig2-MIG2}*Mig2),
        D(mSNF1) ~ -k_{d,mSNF1}*mSNF1 + V_{mSNF1}]

    @named system = ODESystem(equation_system) #Definierar av som är systemet från diffrentialekvationerna
    system = structural_simplify(system) #Skriver om systemet så det blir lösbart

    k_{t,Snf3}= 0.010 #Egentilgen från RGT2!!
    k_{d,Snf3}= 0.231 #Egentilgen från RGT2!!
    k_{t,Std1}= 42.8
    k_{d,Std1}= 0.087
    k_{a,Std1:Rgt1}= #Kommer ändras AAAA
    k_{i,Std1:Rgt1}= #Kommer ändras
    k_{t,Mth1}= 6.000
    k_{d,Mth1}= 0.025
    k_{a,Mth1:Rgt1}= #Kommer ändras AAAA
    k_{i,Mth1:Rgt1}= #Kommer ändras
    k_{t,Rgt1}= 19.000
    k_{d,Rgt1}=0.050

    k_{t,Hxt1}= 1.480
    k_{t,Hxt2}= 4.220
    k_{t,Hxt3}= 4.230
    k_{t,Hxt4}= 1.530
    k_{d,Hxt1}= 0.010
    k_{d,Hxt2}= 0.010
    k_{d,Hxt3}= 0.010
    k_{d,Hxt4}= 0.010
    k_{t,Snf1}= 0.160
    k_{d,Snf1}= 0.020 
    k_{t,Mig1}= 62.000
    k_{d,Mig1}= 0.020
    k_{t,Mig2}= 6.000
    k_{d,Mig2}= 0.046
    V_{transport-Hxt1}= 4.14*10^20
    K_{transport-Hxt1}= 5.40*10^22
    V_{transport-Hxt2}= 5.82*10^19
    K_{transport-Hxt2}= 9.00*10^20
    V_{transport-Hxt3}= 2.16*10^20
    K_{transport-Hxt3}= 3.30*10^22
    V_{transport-Hxt4}= 9.60*10^19
    K_{transport-Hxt4}= 5.58*10^21



    V_{mSNF3}= 50
    θ_{Mig1-Snf3}= 0.000  #Rätt?
    θ_{Mig2-Snf3}= 0.010 
    V_{mSTD1}= 0.040
    θ_{Rgt1_active-Std1}= 0.050
    V_{mMTH1}= 0.170
    θ_{Rgt1_active-MTH1}= 0.030
    θ_{Mig1-MTH1}= 0.460
    θ_{Mig2-MTH1}= 0.001
    V_{mRGT1}= 1.000

    V_{mHXT1}= 2.56
    θ_{Rgt1_active-HXT1}=5.00*10^-001
    V_{mHXT2}= 1.430
    θ_{Rgt1_active-HXT2}= 0.450
    θ_{Mig1-HXT2}= 0.110
    θ_{Mig2-HXT2}= 0.010
    V_{mHXT3}= 2.350
    θ_{Rgt1_active-HXT3}= 0.240
    θ_{Mig1-HXT3}= 0.020
    θ_{Mig2-HXT3}= 0.001
    V_{mHXT4}= 34.200
    θ_{Rgt1_active-HXT4}= 0.026
    θ_{Mig1-HXT4}= 0.430
    θ_{Mig2-HXT4}= 0.080
    V_{mMIG1}= 0.020
    θ_{Mig1-MIG1}= 0.020
    θ_{Mig2-MIG1}= 0.000 #?????
    V_{mMIG2}= 0.230
    θ_{Rgt1_active-MIG2}= 0.100
    θ_{Mig1-MIG2}= 0.001
    θ_{Mig2-MIG2}= 0.010
    V_{mSNF1}= 2.900

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



# Skriv om!!
"Calculate difference between experiments and model"
function cost_function(problem_object, logθ, experimental_data::AbstractVector)
      θ = exp.(logθ)
      error = 0
      for experiment in experimental_data
        # Fixa pre equilibrium!!!!!!!
        #if θ is in
        sol = model_solver(problem_object, θeq, 120) #All have end time 120

        for (index_time, t) in enumerate(experiment.t)
            # Beräkna modellens koncentrationer av HXT generna
            interpolate_point = findfirst(isone,sol.t .> t)
            c_model = interpolate(t,sol.t(interpolate_point),sol.t(interpolate_point-1),sol.u(interpolate_point),sol.u(interpolate_point-1) )

            for (index_hxt, current_hxt_type) in enumerate(data.hxt_types)  #Kika
               error += sum((c_model - data.c[index_time, index_hxt]) .^ 2) #Hitta koncentrationen i c_model som motsvarar rätt HXT-gen
            end
        end
      end
      return error
end

function interpolate(t, t_1, t_2, f_1, f_2)
    return f_1 + (t-t_1) * (f_1 -f_2)/(t_1-t_2)
end


A = [1 2 3
1 2 3]