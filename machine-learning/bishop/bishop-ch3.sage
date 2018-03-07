### Gaussian basis function
x, mu, s = var('x, mu, s')
f(x,mu,s) = exp(-.5*((x-mu)/s)^2)
p0 = plot(f(x,0,1),(x,-3,3),  color='red' )
p1 = plot(f(x,1,1),(x,-3,3),  color='green' )
p2 = plot(f(x,-1,1),(x,-3,3),  color='blue' )
show(p0+p1+p2)

### Sigmoid
f(x) = 1/(1+exp(-x))
p0 = plot(f(x),(x,-10,10),  color='red' )
p1 = plot(f(.5*x), (x,-10,10), color='blue')
p2 = plot(f(-x), (x,-10,10), color='green')
show(p0+p1+p2)

