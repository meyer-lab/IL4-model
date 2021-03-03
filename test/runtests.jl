using IL4model

Ls = 10.0 .^ range(-9, 2, length=50)

outts = [ligOut([1.0, 1.0, ii, 1.0, 10.0, 10.0, 0.1, 10.0])[5] for ii in Ls];
