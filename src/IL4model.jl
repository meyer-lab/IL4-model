module IL4model

using ModelingToolkit, NLsolve


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


function ligOut(ps::Vector; u0 = nothing)::Vector
    @assert all(ps .>= 0.0)
    f2 = (du, u) -> nlsys_func(du, u, ps)

    if u0 === nothing
        u0 = [0.1, 0.3, 0.2]
    end

    res = nlsolve(f2, u0, method = :newton, xtol = 1e-12, ftol = 1e-12)
    if !(res.x_converged | res.f_converged)
        println(res)
    end

    @assert res.x_converged | res.f_converged
    @assert all(res.zero .>= 0.0)
    return res.zero
end

function ligOutLs(Ls::LinRange, ps::Vector)::Matrix
    @assert all(ps .>= 0.0)
    ps = copy(ps)
    pushfirst!(ps, 10 ^ Ls[1])
    results = zeros(length(Ls), length(vars))
    results[1, :] = ligOut(ps)

    for ii in 2:length(Ls)
        ps[1] = 10 ^ Ls[ii]
        results[ii, :] = ligOut(ps; u0 = results[ii - 1, :])
    end

    return results
end

export ligOut, ligOutLs

end # module
