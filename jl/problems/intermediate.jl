using Base

struct MyRange{T<:Int}
    start::T
    step::T
    stop::T

    MyRange(start, step, stop) = start > stop ? error("out of order") : new{Int}(start, step, stop)
end

function _MyRange(r::MyRange, i)
    return r.start + r.step * (i - 1)
end

mr = MyRange(1, 2, 10)

function Base.iterate(mr::MyRange{Int}, state=mr.start)
    if state > mr.stop
        return nothing
    else
        return (state, state + mr.step)
    end
end

for i in MyRange(1, 2, 10)
    println(i)
end

# MyRange(3,1,2)


struct MyLineSpace
    start::Real
    stop::Real
    length::Int
end

ls1 = MyLineSpace(1.2, 2.5, 10)
function Base.iterate(ls::MyLineSpace, state = ls.start)
    if state > ls.stop
        return nothing
    else
        return (state, state + (ls.stop - ls.start) / ls.length)
    end
end

for i in ls1
    println(i)
end
