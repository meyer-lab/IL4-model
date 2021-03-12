using IL4model

Ls = LinRange(-9, 2, 50)

ligOut([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 0.1, 10.0])

ligOutLs(Ls, [1.0, 1.0, 1.0, 10.0, 10.0, 0.1, 10.0])

res = optimizeParam()

println(res)
