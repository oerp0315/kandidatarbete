@parameters t Extracellular_glucose k_{t,Snf3} k_{t,Snf3} k_{a,Snf3} k_{i,Snf3g} k_{t,Std1} k_{d,Std1}  k_{a,Std1:Rgt1} k_{i,Std1:Rgt1} k_{t,Mth1} k_{d,Mth1} k_{a,Mth1:Rgt1} k_{i,Mth1:Rgt1}
@variables  Snf3(t) Snf3g(t) Std1(t) Std1:Rgt1(t) Mth1(t) Mth1:Rgt1(t) Mth1:Rgt1(t) mSNF3(t) mSTD1(t) mMTH1(t) mRGT1(t) #Variabler i modellen
D = Differential(t)

equation_system = [D(Snf3) ~ k_{t,Snf3}*mSNF3- k_{d,Snf3}*Snf3 - k_{a,Snf3}*Snf3*Extracellular_glucose + k_{i,Snf3g}*Snf3g,
    D(Snf3g) ~ k_{a,Snf3}*Snf3*Extracellular_glucose - k_{i,Snf3}*Snf3g,
    D(Std1) ~ k_{t,Std1}*mSTD1 - k_{d,Std1}*Std1- k_{a,Std1:Rgt1}*std1*Rgt1+ k_{i,Std1:Rgt1}*Std1:Rgt1,
    D(Mth1) ~ k_{t,Mth1}*mMTH1 - k_{d,Mth1}*Mth*Snf3g- k_{a,Mth1:Rgt1}*Rgt1*Mth1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1,
    D(Rgt1) ~ k_{t,Rgt1}*mRGT1 - k_{d,Rgt1}*Rgt1 - k_{a,Std1:Rgt1}*Std1*Rgt1 + k_{i,Std1:Rgt1}*Std1:Rgt1 - k_{a,Mth1:Rgt1}*Mth1*Rgt1 + k_{i,Mth1:Rgt1}*Mth1:Rgt1, # Annat utryck om vi testar fosforylering
    D(Std1:Rgt1) ~ k_{a,Std1:Rgt1} *Std1*Rgt1 - k_{i,Std1:Rgt1}*Std1:Rgt1,
    D(Mth1:Rgt1) ~ k_{a,Mth1:Rgt1} *Mth1*Rgt1 - k_{i,Mth1:Rgt1}*Mth1:Rgt1,


    D(mSNF3) ~ -k_{t,mSNF3}*mSNF3 + 1,
    D(mSTD1) ~ -k_{t,mSTD1}*mSTD1 + 1,
    D(mMTH1) ~ -k_{t,mMTH1}*mMTH1 + 1,
    D(mRGT1) ~ -k_{t,mRGT1}*mRGT1 +1]
