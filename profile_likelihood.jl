using Optim

function profile_likelihood(objfun, θ, i; θstep=0.1, Δα1=1e-6, maxsteps=100)
    α0 = objfun(θ)
    Δα = 0
    Δθ = zeros(length(θ))
    Δθ[i] = θstep
    steps = 0

    while steps < maxsteps && abs(Δα) < Δα1
        θnew = θ + Δθ
        θopt = copy(θnew)
        θopt[i] = θ[i]
        objfun_new(θ_vary) = objfun(push!(copy(θ_vary), θ[i]))
        result = optimize(objfun_new, θopt)
        θ[i] = result.minimizer[i]
        Δα = objfun(θ) - α0
        Δθ[i] *= 2
        steps += 1
    end

    return θ, α0, Δα, steps
end

# Define the objective function
function objfun(θ)
    x, y = θ
    return (x - 2)^2 + (y - 3)^2
end

# Set the initial parameter values and the index of the parameter to profile
θ = [0.0, 0.0]
i = 2

# Run the profile likelihood optimization
θ_final, α0, Δα, steps = profile_likelihood(objfun, θ, i)

# Print the results
println("Final parameter values: ", θ_final)
println("Initial objective function value: ", α0)
println("Change in objective function value: ", Δα)
println("Number of steps: ", steps)
