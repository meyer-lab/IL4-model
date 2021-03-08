module IL4model

using ModelingToolkit, NLsolve
import LinearAlgebra: norm


const vars = @variables α Lαβ Lαγ
const params = @parameters L θ Kdα Kdβ Kdγ αT βT γT

#const eqs = [0 ~ Lα*Kdα - α*L,
#             0 ~ Lβ*Kdβ - β*L,
#             0 ~ θ*Kdα*Lαβ - Lβ*α,
#             #0 ~ θ*Kdβ*Lαβ - Lα*β,
#             0 ~ α + Lα + Lαβ + Lαγ - αT,
#             0 ~ Lα*(γT - Lαγ) - θ*Kdγ*Lαγ,
#             0 ~ β + Lβ + Lαβ - βT]

const eqs = [0 ~ θ*Kdα*Lαβ - (((βT - Lαβ) / (1 + L/Kdβ))*L/Kdβ)*α,
             0 ~ α + α*L/Kdα + Lαβ + Lαγ - αT,
             0 ~ (α*L/Kdα)*(γT - Lαγ) - θ*Kdγ*Lαγ]

const ns = NonlinearSystem(eqs, collect(vars), collect(params))
const nlsys_func = @eval eval(generate_function(ns)[2])


function ligOut(ps::Vector{T}; u0 = nothing)::Vector{T} where {T <: Real}
    @assert all(ps .>= 0.0)
    @assert length(ps) == 8
    f2 = (du, u) -> nlsys_func(du, u, ps)

    if u0 === nothing
        u0 = T[0.1, 0.3, 0.2]
    end

    res = nlsolve(f2, u0, method = :newton, xtol = 1e-12, ftol = 1e-12, autodiff = :forward)
    if !(res.x_converged | res.f_converged)
        println(res)
    end

    @assert res.x_converged | res.f_converged
    @assert all(res.zero .>= 0.0)
    return res.zero
end

function ligOutLs(Ls::LinRange, ps::Vector{T})::Matrix{T} where {T <: Real}
    @assert all(ps .>= 0.0)
    @assert length(ps) == 7
    ps = copy(ps)
    pushfirst!(ps, 10 ^ Ls[1])
    results = zeros(T, length(Ls), length(vars))
    results[1, :] = ligOut(ps)

    for ii in 2:length(Ls)
        ps[1] = 10 ^ Ls[ii]
        results[ii, :] = ligOut(ps; u0 = results[ii - 1, :])
    end

    return results
end

""" Calculate the cost of the difference between the model and data. """
function diffcost(ps, c::LinRange, Y; species = 3)::Real
    return norm(ligOutLs(c, ps)[:, species] .- Y)
end

const concs = LinRange(-12.6227, -6.60206, 11)
const IL4sig = [0.1, 0.5, 7.5, 38.1, 82.8, 93.3, 97.1, 94.8, 96.3, 102.0, 101.7]
const neo4sig = [0.7, -0.4, -0.8, -1.0, -0.1, 1.3, 6.7, 14.0, 18.0, 23.0, 24.8]

export ligOut, ligOutLs, concs, IL4sig, neo4sig, diffcost

end # module
