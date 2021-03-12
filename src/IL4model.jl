module IL4model

using ModelingToolkit, NLsolve
import LinearAlgebra: norm
using Optim

const vars = @variables Lαβ Lαγ
const params = @parameters L θ Kdα Kdβ Kdγ αT βT γT

#const eqs = [0 ~ Lα*Kdα - α*L,
#             0 ~ Lβ*Kdβ - β*L,
#             0 ~ θ*Kdα*Lαβ - Lβ*α,
#             #0 ~ θ*Kdβ*Lαβ - Lα*β,
#             0 ~ α + Lα + Lαβ + Lαγ - αT,
#             0 ~ Lα*(γT - Lαγ) - θ*Kdγ*Lαγ,
#             0 ~ β + Lβ + Lαβ - βT]

α = (αT - Lαβ - Lαγ) / (1 + L/Kdα)

const eqs = [0 ~ θ*Kdα*Lαβ - (((βT - Lαβ) / (1 + L/Kdβ))*L/Kdβ)*α,
             0 ~ (α*L/Kdα)*(γT - Lαγ) - θ*Kdγ*Lαγ]

const ns = NonlinearSystem(eqs, collect(vars), collect(params))
const nlsys_func = @eval eval(generate_function(ns)[2])


""" Take in the parameters and calulate the solution for one condition.
    Optionally provide a starting point for solving through u0. """
function ligOut(ps::Vector{T}; u0 = T[0.01, 0.1])::Vector{T} where {T <: Real}
    @assert all(ps .>= 0.0)
    @assert length(ps) == 8
    f2 = (du, u) -> nlsys_func(du, u, ps)

    res = nlsolve(f2, u0, method = :newton, xtol = 1e-12, ftol = 1e-12, autodiff = :forward)
    if !(res.x_converged | res.f_converged)
        println(res)
    end

    @assert res.x_converged | res.f_converged
    @assert all(res.zero .>= 0.0)
    return res.zero
end

""" Calculate a range of solutions. Ls defines the log10 concentration range. """
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

""" Calculate the cost of the difference between the model and data.
    Species 1 is type II signaling, species 2 is type I signaling. """
function diffcost(ps, c::LinRange, Y; species = 2)::Real
    return norm(ligOutLs(c, ps)[:, species] .- Y)
end

""" Fit the two cell lines with two ligand responses. """
function dcost(x)
    x = abs.(copy(x))
    cost = diffcost(x[1:7], concs, IL4sig) + diffcost(x[[1, 8, 9, 10, 5, 6, 7]], concs, neo4sig)
    return cost
end

""" Optimization function. """
function optimizeParam()
    # θ Kdα Kdβ Kdγ αT βT γT
    psInit = [1e6, 1e-8, 1e-9, 1e-6, 1000, 12000, 8000, 1e-8, 1e-9, 1e-6]

    opts = Optim.Options(iterations = 80000,
                         store_trace = false,
                         show_trace = false,
                         f_tol = 1e-11)

    return optimize(dcost, psInit, NewtonTrustRegion(), opts; autodiff = :forward)
end

const concs = LinRange(-12.6227, -6.60206, 11)
const ramosConc = LinRange(-14.2247, -7, 9)
const IL4sig = [0.1, 0.5, 7.5, 38.1, 82.8, 93.3, 97.1, 94.8, 96.3, 102.0, 101.7]
const neo4sig = [0.7, -0.4, -0.8, -1.0, -0.1, 1.3, 6.7, 14.0, 18.0, 23.0, 24.8]
const ramosIL4 = [-0.5, -0.4, 0.5, 14.5, 74.9, 103.8, 107.4, 95.3, 100.8]
const ramosneo4 = [-0.4, -0.4, -0.5, 0.1, 7.8, 61.8, 101.9, 104.1, 102.2]

export ligOut, ligOutLs, concs, IL4sig, neo4sig, diffcost, ramosConc, ramosIL4, ramosneo4, optimizeParam

end # module
