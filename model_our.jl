using DifferentialEquations
using ModelingToolkit
using Plots
using Random
using Distributions
using ForwardDiff
using DataFrames
using CSV
include("newton_minimize.jl")
include("profile_likelihood.jl")

"Object for experimental results"
struct experiment_results
    glucose_conc::Number
    hxt_types::AbstractVector
    c::AbstractMatrix
    t::AbstractVector
end

Data01_glucose = [0.74 0.1 0.06 0.05 0.76 0.13 23.02 26.98
    1.83 0.52 0.1 0.06 0.85 0.33 29.55 36.75
    0.23 0.02 0.29 0.02 0.95 0.05 41.21 53.44
    0.08 0.05 0.19 0.29 0.24 0.14 31.34 44.63
    0.05 0.19 0.15 1.1 0.32 0.18 28.44 25.82
    0.09 0.03 0.06 0.07 0.27 0.09 12.34 16.08
    0.02 0.02 0.01 0.2 0.28 0.04 9.92 13.89]
Data02_glucose = [0.07 0.02 0.06 0.05 0.52 0.13 23.02 26.98
    3.92 1.05 0.09 0.09 0.78 0.71 21.19 24.59
    9.86 8.15 0.18 0.18 0.67 0.57 19.69 20.69
    19.08 15.12 0.32 0.32 1.78 1.46 16.35 17.33
    21.03 20.01 0.35 0.35 3.96 2.96 15.57 14.37
    27.03 24.11 0.91 1.18 7.34 8.34 12.55 13.55
    29.03 31.05 1.03 1.08 11.16 9.16 10.37 8.12]
Data01_mutant = [91.09 81.19
    82.14 74.49
    69.08 68.11
    59.57 64.89
    55.71 56.12
    51.68 60.08]
Data02_mutant = [23.83 20.11
    41.84 34.41
    57.87 51.86
    76.11 65.38
    85.12 77.12
    92.12 81.11]

"Take the avrages of repeted experiments"
function new_data_maker(Data_old)
    size_matrix = size(Data_old)
    println(size_matrix)
    data_new = zeros(size_matrix[1], trunc(Int, size_matrix[2] ./ 2))
    for i = 1:trunc(Int, size_matrix[2] ./ 2)
        global data_new[:, i] = (Data_old[:, 2*i-1] + Data_old[:, 2*i]) / 2
    end
    return data_new
end

# For non avrage data 
#index_general = [1, 2, 3, 4, 5, 6, 7, 8]
#index_mutant = [4, 4]


# For avrage data
index_general = [1, 2, 3, 4]
index_mutant = [4]
Data01_glucose = new_data_maker(Data01_glucose)
Data02_glucose = new_data_maker(Data02_glucose)
Data01_mutant = new_data_maker(Data01_mutant)
Data02_mutant = new_data_maker(Data02_mutant)


timevalues_general = [0.0, 10.0, 20.0, 30.0, 40.0, 60.0, 120.0]
timevalues_mutant = [0.0, 10.0, 27.0, 35.0, 60.0, 120.0]

experiment1 = experiment_results(3.346e8, index_general, Data01_glucose, timevalues_general)
experiment2 = experiment_results(6.685e8, index_general, Data02_glucose, timevalues_general)

experiment3 = experiment_results(3.346e8, index_mutant, Data01_mutant, timevalues_mutant)
experiment4 = experiment_results(3.346e8, index_mutant, Data02_mutant, timevalues_mutant)

#If with mutant data
experimental_data = [experiment1, experiment2, experiment3, experiment4]

#If without mutant data
experimental_data = [experiment1, experiment2]

"Constructs the model with 11 parameters
return a problem_object"
function model_initialize()
    @parameters t Extracellular_glucose k_a_Snf3 k_i_Snf3g k_i_Std1 K_Std1_Rgt1 k_i_Mth1 K_Mth1_Rgt1 k_p_ATP k_i_Snf1 k_i_Mig1 T_mHXT1 θ_activation controller_Rgt1 controller_Mig2
    @variables Snf3(t) Snf3g(t) Std1(t) Mth1(t) Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) mHXT1(t) Hxt1(t) mHXT2(t) Hxt2(t) mHXT3(t) Hxt3(t) mHXT4(t) Hxt4(t) mSNF1(t) Snf1(t) Cellular_glucose(t) mMIG1(t) Mig1(t) mMIG2(t) Mig2(t) Rgt1(t)
    D = Differential(t)

    k_t_Snf3 = 0.010 #From RGT2!!
    k_d_Snf3 = 0.231 #From RGT2!!
    k_t_Std1 = 42.8
    k_d_Std1 = 0.087
    k_t_Mth1 = 6.000
    k_d_Mth1 = 0.025
    k_t_Rgt1 = 19.000
    k_d_Rgt1 = 0.050

    k_t_Hxt1 = 1.480
    k_t_Hxt2 = 4.220
    k_t_Hxt3 = 4.230
    k_t_Hxt4 = 1.530
    k_d_Hxt1 = 0.010
    k_d_Hxt2 = 0.010
    k_d_Hxt3 = 0.010
    k_d_Hxt4 = 0.010
    k_t_Snf1 = 0.160
    k_d_Snf1 = 0.020
    k_t_Mig1 = 62.000
    k_d_Mig1 = 0.020
    k_t_Mig2 = 6.000
    k_d_Mig2 = 0.046
    V_transport_Hxt1 = 4.14 * 10^20
    K_transport_Hxt1 = 5.40 * 10^22
    V_transport_Hxt2 = 5.82 * 10^19
    K_transport_Hxt2 = 9.00 * 10^20
    V_transport_Hxt3 = 2.16 * 10^20
    K_transport_Hxt3 = 3.30 * 10^22
    V_transport_Hxt4 = 9.60 * 10^19
    K_transport_Hxt4 = 5.58 * 10^21


    V_mSNF3 = 50
    θ_Mig1_Snf3 = 0.000  #Not enough decimals
    θ_Mig2_Snf3 = 0.010
    V_mSTD1 = 0.040
    θ_Rgt1_active_Std1 = 0.050
    V_mMTH1 = 0.170
    θ_Rgt1_active_MTH1 = 0.030
    θ_Mig1_MTH1 = 0.460
    θ_Mig2_MTH1 = 0.001
    V_mRGT1 = 1.000

    V_mHXT1 = 2.56
    θ_Rgt1_active_HXT1 = 5.00 * 10^-001
    V_mHXT2 = 1.430
    θ_Rgt1_active_HXT2 = 0.450
    θ_Mig1_HXT2 = 0.110
    θ_Mig2_HXT2 = 0.010
    V_mHXT3 = 2.350
    θ_Rgt1_active_HXT3 = 0.240
    θ_Mig1_HXT3 = 0.020
    θ_Mig2_HXT3 = 0.001
    V_mHXT4 = 34.200
    θ_Rgt1_active_HXT4 = 0.026
    θ_Mig1_HXT4 = 0.430
    θ_Mig2_HXT4 = 0.080

    V_mMIG1 = 0.020
    θ_Mig1_MIG1 = 0.020
    θ_Mig2_MIG1 = 0.000 #Not enough decimals
    V_mMIG2 = 0.230
    θ_Rgt1_active_MIG2 = 0.100
    θ_Mig1_MIG2 = 0.001
    θ_Mig2_MIG2 = 0.010
    V_mSNF1 = 2.900

    k_d_mHXT1 = 0.03
    k_d_mHXT2 = 0.03
    k_d_mHXT3 = 0.03
    k_d_mHXT4 = 0.06
    k_d_mSNF3 = 0.02
    k_d_mMIG1 = 0.04
    k_d_mMIG2 = 0.04
    k_d_mMTH1 = 0.04
    k_d_mSTD1 = 0.01
    k_d_mRGT1 = 0.04
    k_d_mSNF1 = 0.04


    equation_system = [D(Snf3) ~ k_t_Snf3 * mSNF3 - k_d_Snf3 * Snf3 - k_a_Snf3 * Snf3 * Extracellular_glucose + k_i_Snf3g * Snf3g,
        D(Snf3g) ~ k_a_Snf3 * Snf3 * Extracellular_glucose - k_i_Snf3g * Snf3g,
        D(Std1) ~ k_t_Std1 * mSTD1 - k_d_Std1 * Std1 - k_i_Std1 * Std1 * Snf3g,
        D(Mth1) ~ k_t_Mth1 * mMTH1 - k_d_Mth1 * Mth1 - k_i_Mth1 * Mth1 * Snf3g,
        D(Rgt1) ~ k_t_Rgt1 * mRGT1 - k_d_Rgt1 * Rgt1,
        Rgt1_active ~ K_Std1_Rgt1 * Std1 * Rgt1 + K_Mth1_Rgt1 * Mth1 * Rgt1,

        D(Hxt1) ~ k_t_Hxt1 * mHXT1 - k_d_Hxt1 * Hxt1,
        D(Hxt2) ~ k_t_Hxt2 * mHXT2 - k_d_Hxt2 * Hxt2,
        D(Hxt3) ~ k_t_Hxt3 * mHXT3 - k_d_Hxt3 * Hxt3,
        D(Hxt4) ~ k_t_Hxt4 * mHXT4 - k_d_Hxt4 * Hxt4,
        D(Snf1) ~ k_t_Snf1 * mSNF1 - k_d_Snf1 * Snf1 - k_i_Snf1 * Snf1 * Cellular_glucose,
        D(Mig1) ~ k_t_Mig1 * mMIG1 - k_d_Mig1 * Mig1 - k_i_Mig1 * Mig1 * Snf1,
        D(Mig2) ~ k_t_Mig2 * mMIG2 - k_d_Mig2 * Mig2,
        D(Cellular_glucose) ~ V_transport_Hxt1 * Extracellular_glucose / (K_transport_Hxt1 + Extracellular_glucose) + V_transport_Hxt2 * Extracellular_glucose / (K_transport_Hxt2 + Extracellular_glucose) + V_transport_Hxt3 * Extracellular_glucose / (K_transport_Hxt3 + Extracellular_glucose) + V_transport_Hxt4 * Extracellular_glucose / (K_transport_Hxt4 + Extracellular_glucose) - k_p_ATP * Cellular_glucose,

        D(mSNF3) ~ -k_d_mSNF3 * mSNF3 + V_mSNF3 / (1 + θ_Mig1_Snf3 * Mig1) / (1 + θ_Mig2_Snf3 * Mig2),
        D(mSTD1) ~ -k_d_mSTD1 * mSTD1 + V_mSTD1 / (1 + θ_Rgt1_active_Std1 * Rgt1_active),
        D(mMTH1) ~ -k_d_mMTH1 * mMTH1 + V_mMTH1 / (1 + θ_Rgt1_active_MTH1 * Rgt1_active) / (1 + θ_Mig1_MTH1 * Mig1) / (1 + θ_Mig2_MTH1 * Mig2),
        D(mRGT1) ~ controller_Rgt1 * (-k_d_mRGT1 * mRGT1 + V_mRGT1),
        D(mHXT1) ~ -k_d_mHXT1 * mHXT1 + V_mHXT1 * (T_mHXT1 + ((1 - T_mHXT1) * θ_activation * Rgt1) / (1 + θ_activation * Rgt1)) / (1 + θ_Rgt1_active_HXT1 * Rgt1_active), 
        D(mHXT2) ~ -k_d_mHXT2 * mHXT2 + V_mHXT2 / (1 + θ_Rgt1_active_HXT2 * Rgt1_active) / (1 + θ_Mig1_HXT2 * Mig1) / (1 + θ_Mig2_HXT2 * Mig2),
        D(mHXT3) ~ -k_d_mHXT3 * mHXT3 + V_mHXT3 / (1 + θ_Rgt1_active_HXT3 * Rgt1_active) / (1 + θ_Mig1_HXT3 * Mig1) / (1 + θ_Mig2_HXT3 * Mig2),
        D(mHXT4) ~ -k_d_mHXT4 * mHXT4 + V_mHXT4 / (1 + θ_Rgt1_active_HXT4 * Rgt1_active) / (1 + θ_Mig1_HXT4 * Mig1) / (1 + θ_Mig2_HXT4 * Mig2), D(mMIG1) ~ -k_d_mMIG1 * mMIG1 + V_mMIG1 / (1 + θ_Mig1_MIG1 * Mig1) / (1 + θ_Mig2_MIG1 * Mig2),
        D(mMIG2) ~ controller_Mig2 * (-k_d_mMIG2 * mMIG2 + V_mMIG2 / (1 + θ_Rgt1_active_MIG2 * Rgt1_active) / (1 + θ_Mig1_MIG2 * Mig1) / (1 + θ_Mig2_MIG2 * Mig2)),
        D(mSNF1) ~ -k_d_mSNF1 * mSNF1 + V_mSNF1]

    @named system = ODESystem(equation_system)
    system = structural_simplify(system)

    c0 = zeros(24)
    θin = zeros(14)

    u0 = [
        Snf3 => c0[1],
        Snf3g => c0[2],
        Std1 => c0[3],
        Mth1 => c0[4],
        Hxt1 => c0[5],
        Hxt2 => c0[6],
        Hxt3 => c0[7],
        Hxt4 => c0[8],
        Snf1 => c0[9],
        Cellular_glucose => c0[10],
        Mig1 => c0[11],
        Mig2 => c0[12],
        Rgt1 => c0[13],
        mSNF3 => c0[14],
        mSTD1 => c0[15],
        mMTH1 => c0[16],
        mRGT1 => c0[17],
        mHXT1 => c0[18],
        mHXT2 => c0[19],
        mHXT3 => c0[20],
        mHXT4 => c0[21],
        mSNF1 => c0[22],
        mMIG1 => c0[23],
        mMIG2 => c0[24]]

    p = [k_a_Snf3 => θin[1],
        k_i_Snf3g => θin[2],
        k_i_Std1 => θin[3],
        K_Std1_Rgt1 => θin[4],
        k_i_Mth1 => θin[5],
        K_Mth1_Rgt1 => θin[6],
        k_p_ATP => θin[7],
        k_i_Snf1 => θin[8],
        k_i_Mig1 => θin[9],
        T_mHXT1 => θin[10],
        θ_activation => θin[11],
        Extracellular_glucose => θin[12],
        controller_Rgt1 => θin[13],
        controller_Mig2 => θin[14]]

    CSV.write("p_est_results/C_order.csv", DataFrame(index=collect(1:24), C=states(system)))
    CSV.write("p_est_results/param_order.csv", DataFrame(index=collect(1:14), C=parameters(system)))

    tspan = (0.0, 10)
    problem_object = ODEProblem(system, u0, tspan, p)
    return problem_object, system
end

"Constructs the model with 13 parameters.
return a problem_object"
function model_initialize_big()
    @parameters t Extracellular_glucose k_a_Snf3 k_i_Snf3g k_i_Std1 K_Std1_Rgt1 k_i_Mth1 K_Mth1_Rgt1 k_p_ATP k_i_Snf1 k_i_Mig1 T_mHXT1 θ_activation controller_Rgt1 controller_Mig2 θ_Mig1_Snf3 θ_Mig2_MIG1
    @variables Snf3(t) Snf3g(t) Std1(t) Mth1(t) Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) mHXT1(t) Hxt1(t) mHXT2(t) Hxt2(t) mHXT3(t) Hxt3(t) mHXT4(t) Hxt4(t) mSNF1(t) Snf1(t) Cellular_glucose(t) mMIG1(t) Mig1(t) mMIG2(t) Mig2(t) Rgt1(t) #Variabler i modellen
    D = Differential(t)

    k_t_Snf3 = 0.010 #from RGT2!!
    k_d_Snf3 = 0.231 #from RGT2!!
    k_t_Std1 = 42.8
    k_d_Std1 = 0.087
    k_t_Mth1 = 6.000
    k_d_Mth1 = 0.025
    k_t_Rgt1 = 19.000
    k_d_Rgt1 = 0.050

    k_t_Hxt1 = 1.480
    k_t_Hxt2 = 4.220
    k_t_Hxt3 = 4.230
    k_t_Hxt4 = 1.530
    k_d_Hxt1 = 0.010
    k_d_Hxt2 = 0.010
    k_d_Hxt3 = 0.010
    k_d_Hxt4 = 0.010
    k_t_Snf1 = 0.160
    k_d_Snf1 = 0.020
    k_t_Mig1 = 62.000
    k_d_Mig1 = 0.020
    k_t_Mig2 = 6.000
    k_d_Mig2 = 0.046
    V_transport_Hxt1 = 4.14 * 10^20
    K_transport_Hxt1 = 5.40 * 10^22
    V_transport_Hxt2 = 5.82 * 10^19
    K_transport_Hxt2 = 9.00 * 10^20
    V_transport_Hxt3 = 2.16 * 10^20
    K_transport_Hxt3 = 3.30 * 10^22
    V_transport_Hxt4 = 9.60 * 10^19
    K_transport_Hxt4 = 5.58 * 10^21

    
    V_mSNF3 = 50
    #θ_Mig1_Snf3 = 0.000  #Not enough decimals
    θ_Mig2_Snf3 = 0.010
    V_mSTD1 = 0.040
    θ_Rgt1_active_Std1 = 0.050
    V_mMTH1 = 0.170
    θ_Rgt1_active_MTH1 = 0.030
    θ_Mig1_MTH1 = 0.460
    θ_Mig2_MTH1 = 0.001
    V_mRGT1 = 1.000

    V_mHXT1 = 2.56
    θ_Rgt1_active_HXT1 = 5.00 * 10^-001
    V_mHXT2 = 1.430
    θ_Rgt1_active_HXT2 = 0.450
    θ_Mig1_HXT2 = 0.110
    θ_Mig2_HXT2 = 0.010
    V_mHXT3 = 2.350
    θ_Rgt1_active_HXT3 = 0.240
    θ_Mig1_HXT3 = 0.020
    θ_Mig2_HXT3 = 0.001
    V_mHXT4 = 34.200
    θ_Rgt1_active_HXT4 = 0.026
    θ_Mig1_HXT4 = 0.430
    θ_Mig2_HXT4 = 0.080


    V_mMIG1 = 0.020
    θ_Mig1_MIG1 = 0.020
    #θ_Mig2_MIG1 = 0.000 #Not enough decimals
    V_mMIG2 = 0.230
    θ_Rgt1_active_MIG2 = 0.100
    θ_Mig1_MIG2 = 0.001
    θ_Mig2_MIG2 = 0.010
    V_mSNF1 = 2.900

    k_d_mHXT1 = 0.03
    k_d_mHXT2 = 0.03
    k_d_mHXT3 = 0.03
    k_d_mHXT4 = 0.06
    k_d_mSNF3 = 0.02
    k_d_mMIG1 = 0.04
    k_d_mMIG2 = 0.04
    k_d_mMTH1 = 0.04
    k_d_mSTD1 = 0.01
    k_d_mRGT1 = 0.04
    k_d_mSNF1 = 0.04


    equation_system = [D(Snf3) ~ k_t_Snf3 * mSNF3 - k_d_Snf3 * Snf3 - k_a_Snf3 * Snf3 * Extracellular_glucose + k_i_Snf3g * Snf3g,
        D(Snf3g) ~ k_a_Snf3 * Snf3 * Extracellular_glucose - k_i_Snf3g * Snf3g,
        D(Std1) ~ k_t_Std1 * mSTD1 - k_d_Std1 * Std1 - k_i_Std1 * Std1 * Snf3g,
        D(Mth1) ~ k_t_Mth1 * mMTH1 - k_d_Mth1 * Mth1 - k_i_Mth1 * Mth1 * Snf3g,
        D(Rgt1) ~ k_t_Rgt1 * mRGT1 - k_d_Rgt1 * Rgt1,
        Rgt1_active ~ K_Std1_Rgt1 * Std1 * Rgt1 + K_Mth1_Rgt1 * Mth1 * Rgt1,


        D(Hxt1) ~ k_t_Hxt1 * mHXT1 - k_d_Hxt1 * Hxt1,
        D(Hxt2) ~ k_t_Hxt2 * mHXT2 - k_d_Hxt2 * Hxt2,
        D(Hxt3) ~ k_t_Hxt3 * mHXT3 - k_d_Hxt3 * Hxt3,
        D(Hxt4) ~ k_t_Hxt4 * mHXT4 - k_d_Hxt4 * Hxt4,
        D(Snf1) ~ k_t_Snf1 * mSNF1 - k_d_Snf1 * Snf1 - k_i_Snf1 * Snf1 * Cellular_glucose,
        D(Mig1) ~ k_t_Mig1 * mMIG1 - k_d_Mig1 * Mig1 - k_i_Mig1 * Mig1 * Snf1,
        D(Mig2) ~ k_t_Mig2 * mMIG2 - k_d_Mig2 * Mig2,
        D(Cellular_glucose) ~ V_transport_Hxt1 * Extracellular_glucose / (K_transport_Hxt1 + Extracellular_glucose) + V_transport_Hxt2 * Extracellular_glucose / (K_transport_Hxt2 + Extracellular_glucose) + V_transport_Hxt3 * Extracellular_glucose / (K_transport_Hxt3 + Extracellular_glucose) + V_transport_Hxt4 * Extracellular_glucose / (K_transport_Hxt4 + Extracellular_glucose) - k_p_ATP * Cellular_glucose,

        #mRNA
        D(mSNF3) ~ -k_d_mSNF3 * mSNF3 + V_mSNF3 / (1 + θ_Mig1_Snf3 * Mig1) / (1 + θ_Mig2_Snf3 * Mig2),
        D(mSTD1) ~ -k_d_mSTD1 * mSTD1 + V_mSTD1 / (1 + θ_Rgt1_active_Std1 * Rgt1_active),
        D(mMTH1) ~ -k_d_mMTH1 * mMTH1 + V_mMTH1 / (1 + θ_Rgt1_active_MTH1 * Rgt1_active) / (1 + θ_Mig1_MTH1 * Mig1) / (1 + θ_Mig2_MTH1 * Mig2),
        D(mRGT1) ~ controller_Rgt1 * (-k_d_mRGT1 * mRGT1 + V_mRGT1),
        D(mHXT1) ~ -k_d_mHXT1 * mHXT1 + V_mHXT1 * (T_mHXT1 + ((1 - T_mHXT1) * θ_activation * Rgt1) / (1 + θ_activation * Rgt1)) / (1 + θ_Rgt1_active_HXT1 * Rgt1_active),
        D(mHXT2) ~ -k_d_mHXT2 * mHXT2 + V_mHXT2 / (1 + θ_Rgt1_active_HXT2 * Rgt1_active) / (1 + θ_Mig1_HXT2 * Mig1) / (1 + θ_Mig2_HXT2 * Mig2),
        D(mHXT3) ~ -k_d_mHXT3 * mHXT3 + V_mHXT3 / (1 + θ_Rgt1_active_HXT3 * Rgt1_active) / (1 + θ_Mig1_HXT3 * Mig1) / (1 + θ_Mig2_HXT3 * Mig2),
        D(mHXT4) ~ -k_d_mHXT4 * mHXT4 + V_mHXT4 / (1 + θ_Rgt1_active_HXT4 * Rgt1_active) / (1 + θ_Mig1_HXT4 * Mig1) / (1 + θ_Mig2_HXT4 * Mig2), D(mMIG1) ~ -k_d_mMIG1 * mMIG1 + V_mMIG1 / (1 + θ_Mig1_MIG1 * Mig1) / (1 + θ_Mig2_MIG1 * Mig2),
        D(mMIG2) ~ controller_Mig2 * (-k_d_mMIG2 * mMIG2 + V_mMIG2 / (1 + θ_Rgt1_active_MIG2 * Rgt1_active) / (1 + θ_Mig1_MIG2 * Mig1) / (1 + θ_Mig2_MIG2 * Mig2)),
        D(mSNF1) ~ -k_d_mSNF1 * mSNF1 + V_mSNF1]

    @named system = ODESystem(equation_system)
    system = structural_simplify(system)

    c0 = zeros(24)
    θin = zeros(16)

    u0 = [
        Snf3 => c0[1],
        Snf3g => c0[2],
        Std1 => c0[3],
        Mth1 => c0[4],
        Hxt1 => c0[5],
        Hxt2 => c0[6],
        Hxt3 => c0[7],
        Hxt4 => c0[8],
        Snf1 => c0[9],
        Cellular_glucose => c0[10],
        Mig1 => c0[11],
        Mig2 => c0[12],
        Rgt1 => c0[13],
        mSNF3 => c0[14],
        mSTD1 => c0[15],
        mMTH1 => c0[16],
        mRGT1 => c0[17],
        mHXT1 => c0[18],
        mHXT2 => c0[19],
        mHXT3 => c0[20],
        mHXT4 => c0[21],
        mSNF1 => c0[22],
        mMIG1 => c0[23],
        mMIG2 => c0[24]]

    p = [k_a_Snf3 => θin[1],
        k_i_Snf3g => θin[2],
        k_i_Std1 => θin[3],
        K_Std1_Rgt1 => θin[4],
        k_i_Mth1 => θin[5],
        K_Mth1_Rgt1 => θin[6],
        k_p_ATP => θin[7],
        k_i_Snf1 => θin[8],
        k_i_Mig1 => θin[9],
        T_mHXT1 => θin[10],
        θ_activation => θin[11],
        Extracellular_glucose => θin[12],
        controller_Rgt1 => θin[13],
        controller_Mig2 => θin[14],
        θ_Mig1_Snf3 => θin[14],
        θ_Mig2_MIG1 => θin[14]]

    CSV.write("p_est_results/C_order.csv", DataFrame(index=collect(1:24), C=states(system)))
    CSV.write("p_est_results/param_order.csv", DataFrame(index=collect(1:16), C=parameters(system)))

    tspan = (0.0, 10)
    problem_object = ODEProblem(system, u0, tspan, p)
    return problem_object, system
end

"Gives a solution over time 
with points at t_stop_points"
function model_solver(_problem_object::ODEProblem, θin::AbstractVector, c0::AbstractVector, t_stop::Number)
    problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_stop), p=θin)
    sol = solve(problem_object, Rodas5P(), abstol=1e-8, reltol=1e-8; verbose=false)
    return sol
end

"Condition to terminate pre_equilibrium. Happens when gradient is flat enough meaning steady-state is reached"
function terminate_condition(u, t, integrator)
    abstol = 1e-8 / 100
    reltol = 1e-8 / 100

    dudt = DiffEqBase.get_du(integrator)
    valCheck = sqrt(sum((dudt ./ (reltol * integrator.u .+ abstol)) .^ 2) / length(u))
    return valCheck < 1.0
end

"Terminates pre_equilibrium"
function terminate_affect!(integrator)
    terminate!(integrator)
end

"Calculates steady-state concentrations"
function ss_conc_calc(_problem_object::ODEProblem, θin::AbstractVector, c0::AbstractVector)
    t_maximum = 10000 #Maximala tid att nå maximum
    problem_object = remake(_problem_object, u0=convert.(eltype(θin), c0), tspan=(0.0, t_maximum), p=θin)
    cb = DiscreteCallback(terminate_condition, terminate_affect!)
    sol = solve(problem_object, callback=cb, Rodas5P(), abstol=1e-8, reltol=1e-8; verbose=false)
    if sol.retcode ≠ :Success && sol.retcode ≠ :Terminated
        @warn "Failed solving ss ODE, reason: $(sol.retcode)" maxlog = 10
        return Inf
    end
    if sol.t == t_maximum
        @warn "Did not reach steady-state in pre_equilibrium in alloted time" maxlog = 10
        return Inf
    end
    return sol.u[end]
end

"Linear interpolation"
function interpolate(t::Number, f1, f2, t1, t2)
    return (f2 - f1) ./ (t2 - t1) .* (t - t1) + f1
end

"For some error checking in ODE:solver"
function check_extra_error(e)
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
function cost_function(problem_object, logθ, experimental_data::AbstractVector,
    index_first_Hxt=6, index_glucose=3, index_controller_Rgt1=11, index_controller_Mig2=14)

    θ = exp.(logθ)
    θ_type = eltype(θ)

    zero_typefix = convert.(θ_type, 0)
    one_typefix = convert.(θ_type, 1)

    insert!(θ, index_glucose, zero_typefix)
    insert!(θ, index_controller_Rgt1, one_typefix)
    #insert!(θ, index_controller_Mig2, one_typefix)
    append!(θ, one_typefix)

    error = 0
    c_eq_store = []
    for (i, experiment) in enumerate(experimental_data)
        try
            c_eq = [1]
            if i == 2
                c_eq = c_eq_store
            else
                if i == 3
                    θ[index_controller_Rgt1] = zero_typefix
                    θ[index_controller_Mig2] = one_typefix
                else
                    i == 4
                    θ[index_controller_Rgt1] = one_typefix
                    θ[index_controller_Mig2] = zero_typefix
                end

                global c_eq = ss_conc_calc(problem_object, θ, zeros(24))
                if c_eq == Inf
                    return Inf
                end

                if i == 1
                    c_eq_store = c_eq
                end
            end

            if i == 1
                c_eq_store = c_eq
            end

            θ[index_glucose] = convert.(θ_type, experiment.glucose_conc)
            sol = model_solver(problem_object, θ, c_eq, 120) #All experiments have end time 120
            if sol.retcode ≠ :Success
                if sol.retcode ≠ :DtLessThanMin
                    @warn "Failed solving ODE, reason: $(sol.retcode)" maxlog = 10
                end
                return Inf
            end
            for (index_time_data, t) in enumerate(experiment.t)
                index_time_model = convert.(Int64, findfirst(isone, sol.t .>= t))

                if index_time_model == 1
                    c_t = sol.u[1]
                else
                    c_t = interpolate(t, sol.u[index_time_model-1], sol.u[index_time_model], sol.t[index_time_model-1], sol.t[index_time_model])
                end
                for index_hxt = experiment.hxt_types #Kika
                    if i == 3 || i == 4
                        error += sum((c_t[index_first_Hxt-1+4] - experiment.c[index_time_data, 1]) .^ 2)
                    else
                        #For without mutant
                        #error += sum((c_t[index_first_Hxt-1+index_hxt] - experiment.c[index_time_data, index_hxt]) .^ 2) #Håll koll på så index (+5 blir rätt)
                        
                        #For with mutant
                        error += sum((c_t[index_first_Hxt-1+index_hxt] - experiment.c[index_time_data, ceil(Int, index_hxt / 2)]) .^ 2) #Håll koll på så index (+5 blir rätt)
                    end
                end
            end
        catch e 
            check_extra_error(e)
            return Inf
        end
    end
    return error
end

function timing_tests(problem_object, experimental_data, f)
    #Solve one time first to fix compliation time
    model_solver(problem_object, ones(12), zeros(26), 100)
    cost_function(problem_object, zeros(11), experimental_data)
    ForwardDiff.gradient(f, ones(11))
    ForwardDiff.hessian(f, ones(11))

    time_model_solver = @elapsed model_solver(problem_object, ones(12), zeros(26), 100)
    time_cost_function = @elapsed cost_function(problem_object, zeros(11), experimental_data)
    time_gradient = @elapsed ForwardDiff.gradient(f, ones(11))
    time_hessian = @elapsed ForwardDiff.hessian(f, ones(11))
    data = DataFrame(Function=["model_solver", "cost_function", "gradient", "hessian"], time=[time_model_solver, time_cost_function, time_gradient, time_hessian])
    CSV.write("ss_timer.csv", data)
end

"Timing operations for big modell"
function timing_tests_big(problem_object, experimental_data, f)
    #Solve one time first to fix compliation time
    model_solver(problem_object, ones(16), zeros(26), 100)
    cost_function(problem_object, zeros(13), experimental_data)
    ForwardDiff.gradient(f, ones(13))
    ForwardDiff.hessian(f, ones(13))

    time_model_solver = @elapsed model_solver(problem_object, ones(16), zeros(26), 100)
    time_cost_function = @elapsed cost_function(problem_object, zeros(13), experimental_data)
    time_gradient = @elapsed ForwardDiff.gradient(f, ones(13))
    time_hessian = @elapsed ForwardDiff.hessian(f, ones(13))
    data = DataFrame(Function=["model_solver", "cost_function", "gradient", "hessian"], time=[time_model_solver, time_cost_function, time_gradient, time_hessian])
    println(time_model_solver)
    println(time_cost_function)
    println(time_gradient)
    println(time_hessian)
    CSV.write("ss_timer.csv", data)
end

function bounds_generator(θ_estimation)
    #bounds = [(1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3)]
    bounds = [(1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4)]
    newbounds = bounds
    for i = 1:11
        newbounds[i] = θ_estimation[i] .* bounds[i]
    end
    return newbounds
end

function bounds_generator_big(θ_estimation)
    bounds = [(1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3), (1e-3, 1e3)]
    #bounds = [(1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4), (1e-4, 1e4)]
    newbounds = bounds
    for i = 1:13
        newbounds[i] = θ_estimation[i] .* bounds[i]
    end
    return newbounds
end

"Plot ode and compare with data"
function plot_kinetic(_θ,experimental_data,
    index_first_Hxt=6, index_glucose=3, index_controller_Rgt1=11, index_controller_Mig2=14)
    θ = _θ
    θ_type = eltype(θ)

    zero_typefix = convert.(θ_type, 0)
    one_typefix = convert.(θ_type, 1)

    insert!(θ, index_glucose, zero_typefix)
    insert!(θ, index_controller_Rgt1, one_typefix)
    insert!(θ, index_controller_Mig2, one_typefix)
    

    c_eq_store = []
    for (i, experiment) in enumerate(experimental_data)
        c_eq = [1]
        if i == 2
            c_eq = c_eq_store
        else
            if i == 3
                θ[index_controller_Rgt1] = zero_typefix
                θ[index_controller_Mig2] = one_typefix
            else
                i == 4
                θ[index_controller_Rgt1] = one_typefix
                θ[index_controller_Mig2] = zero_typefix
            end

            global c_eq = ss_conc_calc(problem_object, θ, zeros(24))
            if c_eq == Inf
                return Inf
            end

            if i == 1
                c_eq_store = c_eq
            end
        end
        θ[index_glucose] = convert.(θ_type, experiment.glucose_conc)
        sol = model_solver(problem_object, θ, c_eq, 120) #All experiments have end time 120
        if sol.retcode ≠ :Success
            if sol.retcode ≠ :DtLessThanMin
                @warn "Failed solving ODE, reason: $(sol.retcode)" maxlog = 10
            end
            return Inf
        end

        model_conc = transpose(Matrix(sol))
        if i == 1 || i==2
            plot(sol.t, model_conc[:,18], labels ="Modell mHXT1", linewidth = 2, c=RGB(0.41, 0.82, 0.91), ylims=(0,22))
            plot!(sol.t, model_conc[:,19], labels ="Modell mHXT2", linewidth = 2, c=RGB(0.98, 0.41, 0))
            plot!(sol.t, model_conc[:,20], labels ="Modell mHXT3", linewidth = 2, c=RGB(177/255, 58/255, 105/255))
            plot!(sol.t, model_conc[:,21], labels ="Modell mHXT4", linewidth = 2, c=RGB(102/255, 188/255, 102/255))
            
            plot!(experiment.t, experiment.c[:,1], labels = "Data mHXT1", seriestype=:scatter, c=RGB(0.41, 0.82, 0.91))
            plot!(experiment.t, experiment.c[:,2], labels = "Data mHXT2", seriestype=:scatter, c=RGB(0.98, 0.41, 0))
            plot!(experiment.t, experiment.c[:,3], labels = "Data  mHXT3", seriestype=:scatter, c=RGB(177/255, 58/255, 105/255))
            plot!(experiment.t, experiment.c[:,4], labels = "Data  mHXT4", seriestype=:scatter, c=RGB(102/255, 188/255, 102/255))
        else
            plot(sol.t, model_conc[:,21], labels ="mHXT4", linewidth = 2, c=RGB(102/255, 188/255, 102/255), ylims=(0,80))
            plot!(experiment.t, experiment.c, labels = ["mHXT4" "mHXT4"], seriestype=:scatter, c=RGB(102/255, 188/255, 102/255))
        end
        plot!(xlabel="Tid (s)", ylabel="Koncentration (molekyler/cell)", legend=:topright)
        savefig("p_est_results/plot_ode_over_t$i")
    end
end

# Avrages + Mutant + Big modell (-3, 3)*short_optim bounds
long_optim = [0.7969990063233143, 68.93424287588466, 6997.780815301634, 0.01593630757371225, 17.286346860335527, 0.008671320875673462, 0.00012446646232616555, 304.4640461729089, 111.99828602219864, 0.00030830983134835766, 235.26237907052467, 0.0001357129836595192, 23973.609058384074]


# 11 parameter modell
#=
problem_object, system = model_initialize()
bounds = bounds_generator(ones(11))
log_bounds = map(x -> (log(x[1]), log(x[2])), bounds)
f(x) = cost_function(problem_object, x, experimental_data) # 3 är index för glukos
timing_tests(problem_object, experimental_data, f) # run the parameter estimation
time = @elapsed x_min, f_min = p_est(f, log_bounds, 1000, false)
println("The optimization took: $time")
plot_kinetic(long_optim, experimental_data)
=#


# 13 parameter modell
problem_object, system = model_initialize_big()
bounds = bounds_generator_big(ones(13))
log_bounds = map(x -> (log(x[1]), log(x[2])), bounds)
f(x) = cost_function(problem_object, x, experimental_data, 18, 3, 12, 16) # 3 är index för glukos
timing_tests_big(problem_object, experimental_data, f)
time = @elapsed x_min, f_min = p_est(f, log_bounds, 1000, false) # run the parameter estimation
println("The optimization took: $time")
plot_kinetic(long_optim, experimental_data, 18, 3, 12, 16)








#=
# Define the initial parameter values
params = x_min

# Perform profile likelihood analysis for each parameter
num_points = 100
threshold = 3.84

# save threshold
CSV.write("profilelikelihood_results/threshold.csv", DataFrame(threshold=threshold))

run_profile_likelihood(params, log_bounds, 50, num_points, threshold)