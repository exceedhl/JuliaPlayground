using Test

function transform_digit_to_bitvec(digit, levels)
    vec = zeros(levels)
    setindex!(vec, 1, digit + 1)
    return vec
end

function transform_bitvec_to_digit(vec)
    findmax(vec)[2] - 1
end

function array_to_vector(arr)
    mapreduce(x -> reshape(x, length(x)), vcat, arr, init = Float64[])
end

function vector_to_array(v, dims)
    result = []
    i = 1
    for dim in dims
        push!(result, reshape(v[i:i+dim[1]*dim[2]-1], dim...))
        i += prod(dim)
    end
    result
end

#test
init_θ1 = rand(4,3)
init_θ2 = rand(5,4)
@test array_to_vector([init_θ1, init_θ2]) ==
    array_to_vector(vector_to_array(array_to_vector([init_θ1, init_θ2]), [(4,3), (5,4)]))
