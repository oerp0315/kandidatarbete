@parameters t θ[1:4]     #Parametrar i modellen
@variables c1(t) c2(t) c3(t) Snf3(t)   #Variabler i modellen
D = Differential(t) #Definierar tecken för derivata

equation_system = [D(c1) ~ -θ[1]*c1+θ[2]*c2,
            D(c2) ~ θ[1]*c1-θ[2]*c2-θ[3]*c2+θ[4]*c3,
            D(c3) ~ θ[3]*c2-θ[4]*c3]
