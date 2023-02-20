using Random
using Plots

function latin_square(n::Int)
    if n < 2
        error("n must be at least 2")
    end

    # Initialize the Latin square as an n-by-n array of zeros
    square = zeros(Int, n, n)

    # Fill the first row with random integers between 1 and n
    square[1, :] = randperm(n)

    # Fill the remaining rows with shifted copies of the first row
    for i in 2:n
        square[i, :] = circshift(square[i-1, :], 1)
    end

    # Convert the Latin square to a matrix of values between 0 and 1
    samples = zeros(n, n)
    for i in 1:n
        for j in 1:n
            samples[i, j] = (square[i, j] - 1) / n
        end
    end

    return samples
end

function plot_latin_square(n::Int)
    samples = latin_square(n)
    x = samples[:, 1]
    y = samples[:, 2]
    z = rand(length(x))
    scatter(x, y, zcolor=z, legend=false)
    xlabel!("x")
    ylabel!("y")
    zlabel!("z")
end

plot_latin_square(16)
