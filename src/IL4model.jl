module IL4model

using ModelingToolkit, NLsolve, StatsFuns


@variables α β Lα Lβ Lαβ Lαγ
@parameters Kdα Kdβ L θ αT γT Kdγ βT

const eqs = [0 ~ Lα*Kdα - α*L,
             0 ~ Lβ*Kdβ - β*L,
             0 ~ θ*Kdα*Lαβ - Lβ*α,
             #0 ~ θ*Kdβ*Lαβ - Lα*β,
             0 ~ α + Lα + Lαβ + Lαγ - αT,
             0 ~ Lα*(γT - Lαγ) - θ*Kdγ*Lαγ,
             0 ~ β + Lβ + Lαβ - βT]

# Just type 1
const t1eqs = [0 ~ L * (αT - Lαγ) * (γT - Lαγ) - θ * Kdγ * (Kdα + L)]

# Just type I: L * (αT - Lαγ) * (γT - Lαγ) = θ * Kdγ * (Kdα + L)

const ns = NonlinearSystem(eqs, [α, β, Lα, Lβ, Lαβ, Lαγ], [Kdα, Kdβ, L, θ, αT, γT, Kdγ, βT])
const nlsys_func = @eval eval(generate_function(ns)[2])

const nst1 = NonlinearSystem(eqs, [α, Lαγ], [Kdα, L, θ, αT, γT, Kdγ])
const nlsyst1_func = @eval eval(generate_function(nst1)[2])

function ligOut(ps::Vector)::Vector
    @assert all(ps .>= 0.0)
    f2 = (du, u) -> nlsys_func(du, softplus.(u), ps)

    res = nlsolve(f2, zeros(6), method = :newton, xtol = 1e-12, ftol = 1e-12)
    if !(res.x_converged | res.f_converged)
        println(res)
    end

    @assert res.x_converged | res.f_converged
    return softplus.(res.zero)
end

export ligOut

end # module
