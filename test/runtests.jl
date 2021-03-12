using IL4model
using Optim

ligOut([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 0.1, 10.0])
ligOutLs(concs, [1.0, 1.0, 1.0, 10.0, 10.0, 0.1, 10.0])

res = optimizeParam()

@assert Optim.minimum(res) < 100.0
