### Plot f1-score function
p,r = var('p,r')
f1(p,r) = 2 /(1/p+1/r)
d = contour_plot(f1, (0,1), (0,1))
