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
    @parameters t Extracellular_glucose k_a_Snf3 k_i_Snf3g k_i_Std1 K_Std1_Rgt1 k_i_Mth1 K_Mth1_Rgt1 k_p_ATP k_i_Snf1 k_i_Mig1 T_mHXT1 θ_activation
    @variables Snf3(t) Snf3g(t) Std1(t) Mth1(t) Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) mHXT1(t) Hxt1(t) mHXT2(t) Hxt2(t) mHXT3(t) Hxt3(t) mHXT4(t) Hxt4(t)  mSNF1(t) Snf1(t) Cellular_glucose(t) mMIG1(t) Mig1(t) mMIG2(t) Mig2(t) Rgt1(t) #Variabler i modellen
    D = Differential(t)

    k_t_Snf3= 0.010 #Egentilgen från RGT2!!
    k_d_Snf3= 0.231 #Egentilgen från RGT2!!
    k_t_Std1= 42.8
    k_d_Std1= 0.087
    k_t_Mth1= 6.000
    k_d_Mth1= 0.025
    k_t_Rgt1= 19.000
    k_d_Rgt1=0.050

    k_t_Hxt1= 1.480
    k_t_Hxt2= 4.220
    k_t_Hxt3= 4.230
    k_t_Hxt4= 1.530
    k_d_Hxt1= 0.010
    k_d_Hxt2= 0.010
    k_d_Hxt3= 0.010
    k_d_Hxt4= 0.010
    k_t_Snf1= 0.160
    k_d_Snf1= 0.020 
    k_t_Mig1= 62.000
    k_d_Mig1= 0.020
    k_t_Mig2= 6.000
    k_d_Mig2= 0.046
    V_transport_Hxt1= 4.14*10^20
    K_transport_Hxt1= 5.40*10^22
    V_transport_Hxt2= 5.82*10^19
    K_transport_Hxt2= 9.00*10^20
    V_transport_Hxt3= 2.16*10^20
    K_transport_Hxt3= 3.30*10^22
    V_transport_Hxt4= 9.60*10^19
    K_transport_Hxt4= 5.58*10^21

    V_mSNF3= 50
    θ_Mig1_Snf3= 0.000  #Rätt?
    θ_Mig2_Snf3= 0.010 
    V_mSTD1= 0.040
    θ_Rgt1_active_Std1= 0.050
    V_mMTH1= 0.170
    θ_Rgt1_active_MTH1= 0.030
    θ_Mig1_MTH1= 0.460
    θ_Mig2_MTH1= 0.001
    V_mRGT1= 1.000

    V_mHXT1= 2.56
    θ_Rgt1_active_HXT1=5.00*10^-001
    V_mHXT2= 1.430
    θ_Rgt1_active_HXT2= 0.450
    θ_Mig1_HXT2= 0.110
    θ_Mig2_HXT2= 0.010
    V_mHXT3= 2.350
    θ_Rgt1_active_HXT3= 0.240
    θ_Mig1_HXT3= 0.020
    θ_Mig2_HXT3= 0.001
    V_mHXT4= 34.200
    θ_Rgt1_active_HXT4= 0.026
    θ_Mig1_HXT4= 0.430
    θ_Mig2_HXT4= 0.080
    V_mMIG1= 0.020
    θ_Mig1_MIG1= 0.020
    θ_Mig2_MIG1= 0.000 #?????
    V_mMIG2= 0.230
    θ_Rgt1_active_MIG2= 0.100
    θ_Mig1_MIG2= 0.001
    θ_Mig2_MIG2= 0.010
    V_mSNF1= 2.900


    k_d_mHXT1=0
    k_d_mHXT2=0
    k_d_mHXT3=0
    k_d_mHXT4=0
    k_d_mSNF3=0
    k_d_mMIG1=0
    k_d_mMIG2=0
    k_d_mMTH1=0
    k_d_mSTD1=0
    k_d_mRGT1=0
    k_d_mSNF1=0


    equation_system = [D(Snf3) ~ k_t_Snf3*mSNF3- k_d_Snf3*Snf3 - k_a_Snf3*Snf3*Extracellular_glucose + k_i_Snf3g*Snf3g,
        D(Snf3g) ~ k_a_Snf3*Snf3*Extracellular_glucose - k_i_Snf3g*Snf3g,
        D(Std1) ~ k_t_Std1*mSTD1 - k_d_Std1*Std1 - k_i_Std1*Std1*Snf3g,
        D(Mth1) ~ k_t_Mth1*mMTH1 - k_d_Mth1*Mth1 - k_i_Mth1*Mth1*Snf3g,
        D(Rgt1) ~ k_t_Rgt1*mRGT1 - k_d_Rgt1*Rgt1, # Annat utryck om vi testar fosforylering
        Rgt1_active ~ K_Std1_Rgt1*Std1*Rgt1 + K_Mth1_Rgt1*Mth1*Rgt1,

        # Kinetik om vi antar mindre steady-state
        #D(Snf3) ~ k_t_Snf3*mSNF3- k_d_Snf3*Snf3 - k_a_Snf3*Snf3*Extracellular_glucose + k_i_Snf3g*Snf3g,
        #D(Std1) ~ k_t_Std1*mSTD1 - k_d_Std1*Std1 - k_i_Std1*Std1*Snf3g- k_a_Std1_Rgt1*std1*Rgt1+ k_i_Std1_Rgt1*Std1_Rgt1,
        #D(Mth1) ~ k_t_Mth1*mMTH1 - k_d_Mth1*Mth1 - k_i_Mth1*Mth1*Snf3g- k_a_Mth1_Rgt1*Rgt1*Mth1 + k_i_Mth1_Rgt1*Mth1_Rgt1,
        #D(Rgt1) ~ k_t_Rgt1*mRGT1 - k_d_Rgt1*Rgt1 - k_a_Std1_Rgt1*Std1*Rgt1 + k_i_Std1_Rgt1*Std1_Rgt1 - k_a_Mth1_Rgt1*Mth1*Rgt1 + k_i_Mth1_Rgt1*Mth1_Rgt1, # Annat utryck om vi testar fosforylering
        #D(Std1_Rgt1) ~ k_a_Std1_Rgt1 *Std1*Rgt1 - k_i_Std1_Rgt1*Std1_Rgt1, 
        #D(Mth1_Rgt1) ~ k_a_Mth1_Rgt1 *Mth1*Rgt1 - k_i_Mth1_Rgt1*Mth1_Rgt1,
        #Rgt1_active ~ Std1_Rgt1 + Mth1_Rgt1_

        D(Hxt1)~k_t_Hxt1*mHXT1 - k_d_Hxt1*Hxt1, 
        D(Hxt2)~k_t_Hxt2*mHXT2 - k_d_Hxt2*Hxt2, 
        D(Hxt3)~k_t_Hxt3*mHXT3 - k_d_Hxt3*Hxt3,
        D(Hxt4)~k_t_Hxt4*mHXT4 - k_d_Hxt4*Hxt4,
        D(Snf1)~ k_t_Snf1*mSNF1 - k_d_Snf1*Snf1 + k_i_Snf1*Snf1*Cellular_glucose,
        D(Mig1)~ k_t_Mig1*mMIG1 - k_d_Mig1*Mig1 -  k_i_Mig1*Mig1*Snf1,
        D(Mig2)~k_t_Mig2*mMIG2 - k_d_Mig2*Mig2,
        D(Cellular_glucose)~ V_transport_Hxt1*Extracellular_glucose/(K_transport_Hxt1 + Extracellular_glucose) + V_transport_Hxt2*Extracellular_glucose/(K_transport_Hxt2 + Extracellular_glucose) + V_transport_Hxt3*Extracellular_glucose/(K_transport_Hxt3 + Extracellular_glucose) + V_transport_Hxt4*Extracellular_glucose/(K_transport_Hxt4 + Extracellular_glucose) - k_p_ATP*Cellular_glucose, 

        #mRNA
        D(mSNF3) ~ -k_d_mSNF3*mSNF3 + V_mSNF3/(1+θ_Mig1_Snf3*Mig1)/(1+θ_Mig2_Snf3*Mig2),
        D(mSTD1) ~ -k_d_mSTD1*mSTD1 + V_mSTD1/(1+θ_Rgt1_active_Std1*Rgt1_active),
        D(mMTH1) ~ -k_d_mMTH1*mMTH1 + V_mMTH1/(1+θ_Rgt1_active_MTH1*Rgt1_active)/(1+θ_Mig1_MTH1*Mig1)/(1+θ_Mig2_MTH1*Mig2),
        D(mRGT1) ~ - k_d_mRGT1*mRGT1 + V_mRGT1,

        D(mHXT1) ~ -k_d_mHXT1*mHXT1 + V_mHXT1*(T_mHXT1+((1-T_mHXT1)*θ_activation*Rgt1)/(1+θ_activation*Rgt1))/(1+θ_Rgt1_active_HXT1*Rgt1_active), # Vi har tagit bort glucose signals effekt. Läs på om basalreguleringen
        D(mHXT2) ~ - k_d_mHXT2*mHXT2+ V_mHXT2/(1+θ_Rgt1_active_HXT2*Rgt1_active)/(1+θ_Mig1_HXT2*Mig1)/(1+θ_Mig2_HXT2*Mig2),
        D(mHXT3) ~ -k_d_mHXT3*mHXT3 + V_mHXT3/(1+θ_Rgt1_active_HXT3*Rgt1_active)/(1+θ_Mig1_HXT3*Mig1)/(1+θ_Mig2_HXT3*Mig2),
        D(mHXT4) ~ -k_d_mHXT4*mHXT4 + V_mHXT4/(1+θ_Rgt1_active_HXT4*Rgt1_active)/(1+θ_Mig1_HXT4*Mig1)/(1+θ_Mig2_HXT4*Mig2),
        D(mMIG1) ~ -k_d_mMIG1*mMIG1 + V_mMIG1/(1+θ_Mig1_MIG1*Mig1)/(1+θ_Mig2_MIG1*Mig2),
        D(mMIG2) ~ -k_d_mMIG2*mMIG2 + V_mMIG2/(1+θ_Rgt1_active_MIG2*Rgt1_active)/(1+θ_Mig1_MIG2*Mig1)/(1+θ_Mig2_MIG2*Mig2),
        D(mSNF1) ~ -k_d_mSNF1*mSNF1 + V_mSNF1]

    @named system = ODESystem(equation_system) #Definierar av som är systemet från diffrentialekvationerna
    system = structural_simplify(system) #Skriver om systemet så det blir lösbart


    # Intialvärden som kommer skrivas över
    c0 = zeros(26)
    θin = zeros(12)

    u0 = [Extracellular_glucose => c0[1],
        Snf3 => c0[2],
        Snf3g => c0[3],
        Std1 => c0[4],
        Mth1 => c0[5],
        Rgt1_active => c0[6],
        Hxt1 => c0[7],
        Hxt2 => c0[8],
        Hxt3 => c0[9],
        Hxt4 => c0[10],
        Snf1 => c0[11],
        Cellular_glucose => c0[12],
        Mig1 => c0[13],
        Mig2 => c0[14],
        Rgt1 => c0[15],
        
        mSNF3 => c0[16],
        mSTD1 => c0[17],
        mMTH1 => c0[18],
        mRGT1 => c0[19],
        mHXT1 => c0[20],
        mHXT2 => c0[21],
        mHXT3 => c0[22],
        mHXT4 => c0[23],
        mSNF1 => c0[24],
        mMIG1 => c0[25],
        mMIG2 => c0[26]]

    p = [Extracellular_glucose => θin[1],
        k_a_Snf3 => θin[2],
        k_i_Snf3g => θin[3],
        k_i_Std1 => θin[4],
        K_Std1_Rgt1 => θin[5],
        k_i_Mth1 => θin[6], 
        K_Mth1_Rgt1 => θin[7],

        k_p_ATP => θin[8], 
        k_i_Snf1 => θin[9],
        k_i_Mig1 => θin[10], 

        T_mHXT1 => θin[11],
        θ_activation => θin[12] ]


      tspan = (0.0, 10) #Tiden vi kör modellen under
      problem_object = ODEProblem(system, u0, tspan, p, jac=true)  #Definierar vad som ska beräknas
      return problem_object
end

function model_solver(_problem_object, θin, c0, t_stop)
    problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_stop), p=θin)
    solution = solve(_problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8)
    return solution
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

c0 = zeros(26)
θin = zeros(12)
problem_object = model_initialize()
#solution = model_solver(problem_object, θin, c0, 20)

problem_object = remake(problem_object,u0=zeros(26))
#problem_object = remake(problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, 10), p=θin)
solution = solve(problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8)
#plot(solution)