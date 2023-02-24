using Random
using Plots

function latin_hypercube(n_samples, bounds; seed=123)
    Random.seed!(seed)

    if n_samples < 2
        error("n must be at least 2")
    end

    n_vars = length(bounds)

    # Initialize the Latin square as an n-by-n array of zeros
    square = zeros(Int, n_samples, n_samples)

    # Fill the first row with random integers between 1 and n
    square[1, :] = randperm(n_samples)

    # Fill the remaining rows with shifted copies of the first row
    for i in 2:n_samples
        square[i, :] = circshift(square[i-1, :], 1)
    end

    # create random values added to sample values
    random_matrix = rand(n_samples, n_vars)

    # create a matrix where samples will be inserted to
    samples = zeros(n_samples, n_vars)

    # generate samples with random position within varible intervals
    for i in 1:n_samples
        for j in 1:n_vars
            samples[i, j] = (square[i, j] - 1) / ((n_samples - 1) * (n_samples / (n_samples - 1))) + random_matrix[i, j] / n_samples
        end
    end

    # scale samples to bounds
    for i in 1:n_samples
        for j in 1:n_vars
            samples[i, j] = (bounds[j][2] - bounds[j][1]) * samples[i, j] + bounds[j][1]
        end
    end

    return samples
end

function plot_latin_hypercube(n_samples, bounds)
    samples = latin_hypercube(n_samples, bounds)
    x = samples[:, 1]
    y = samples[:, 2]
    z = rand(length(x))
    scatter(x, y, zcolor=z, legend=false)
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")
end

bounds = [(-1, 1), (-4, 4)]

plot_latin_hypercube(16, bounds)
