function model_initialize()
    @parameters t Extracellular_glucose k_{t,Snf3} k_{t,Snf3} k_{a,Snf3} k_{i,Snf3g} k_{t,Std1} k_{d,Std1}  k_{a,Std1:Rgt1} k_{i,Std1:Rgt1} k_{t,Mth1} k_{d,Mth1} k_{a,Mth1:Rgt1} k_{i,Mth1:Rgt1}  V_{mSNF3} θ_{Mig1-Snf3} θ_{Mig2-Snf3} V_{mSTD1} θ_{Rgt1_active-Snf3} V_{mMTH1} θ_{Rgt1_active-MTH1} θ_{Mig1-MTH1} θ_{Mig2-MTH1} V_{mRGT1}
    @variables  Snf3(t) Snf3g(t) Std1(t) Std1:Rgt1(t) Mth1(t) Mth1:Rgt1(t) Mth1:Rgt1(t)  Rgt1_active(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) #Variabler i modellen
    D = Differential(t)

    equation_system = [D(Snf3) ~ k_{t,Snf3}*mSNF3- k_{d,Snf3}*Snf3 - k_{a,Snf3}*Snf3*Extracellular_glucose + k_{i,Snf3g}*Snf3g,
        D(Snf3g) ~ k_{a,Snf3}*Snf3*Extracellular_glucose - k_{i,Snf3}*Snf3g,
        D(Std1) ~ k_{t,Std1}*mSTD1 - k_{d,Std1}*Std1- k_{a,Std1:Rgt1}*std1*Rgt1+ k_{i,Std1:Rgt1}*Std1:Rgt1,
        D(Mth1) ~ k_{t,Mth1}*mMTH1 - k_{d,Mth1}*Mth*Snf3g- k_{a,Mth1:Rgt1}*Rgt1*Mth1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1,
        D(Rgt1) ~ k_{t,Rgt1}*mRGT1 - k_{d,Rgt1}*Rgt1 - k_{a,Std1:Rgt1}*Std1*Rgt1 + k_{i,Std1:Rgt1}*Std1:Rgt1 - k_{a,Mth1:Rgt1}*Mth1*Rgt1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1, # Annat utryck om vi testar fosforylering
        D(Std1:Rgt1) ~ k_{a,Std1:Rgt1} *Std1*Rgt1 - k_{i,Std1:Rgt1}*Std1:Rgt1,
        D(Mth1:Rgt1) ~ k_{a,Mth1:Rgt1} *Mth1*Rgt1 - k_{i,Mth1:Rgt1}*Mth1:Rgt1,

        Rgt1_active ~ Std1:Rgt1 + Mth1:Rgt1,

        D(mSNF3) ~ -k_{t,mSNF3}*mSNF3 + V_{mSNF3}/(1+θ_{Mig1-Snf3}*Mig1)/(1+θ_{Mig2-Snf3}*Mig2),
        D(mSTD1) ~ -k_{t,mSTD1}*mSTD1 + V_{mSTD1}/(1+θ_{Rgt1_active-Snf3}*Rgt1_active),
        D(mMTH1) ~ -k_{t,mMTH1}*mMTH1 + V_{mMTH1}/(1+θ_{Rgt1_active-MTH1}*Rgt1_active)/(1+θ_{Mig1-MTH1}*Mig1)/(1+θ_{Mig2-MTH1}*Mig2),
        D(mRGT1) ~ -k_{t,mRGT1}*mRGT1 +V_{mRGT1}]

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