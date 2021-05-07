module IL4model

import LinearAlgebra: norm
using Optim
using polyBindingModel


""" Take in the parameters and calulate the solution for one condition. """
function ligOut(ps::Vector{T})::T where {T <: Real}
    @assert all(ps .>= 0.0)
    @assert length(ps) == 10
    
    # TODO: Add polyc here.
    # ps: concentration, receptor affinities (3), receptor amounts (3), Kx, pSTAT weight, scaling

    cplx = [0.0, 0.0]
    return p[10] * (cplx[1] + p[9] * cplx[2])
end

""" Calculate a range of solutions. Ls defines the log10 concentration range. """
function ligOutLs(Ls::LinRange, ps::Vector{T})::Vector{T} where {T <: Real}
    @assert all(ps .>= 0.0)
    @assert length(ps) == 7
    ps = copy(ps)
    pushfirst!(ps, 0.0)
    results = zeros(T, length(Ls))

    for ii in 1:length(Ls)
        ps[1] = 10 ^ Ls[ii]
        results[ii] = ligOut(ps)
    end

    return results
end

""" Calculate the cost of the difference between the model and data.
    Species 1 is type II signaling, species 2 is type I signaling. """
function diffcost(ps, c::LinRange, Y)::Real
    return norm(ligOutLs(c, ps) .- Y)
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
