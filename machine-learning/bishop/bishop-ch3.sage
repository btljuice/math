random_seed = 42

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


### Plot 24 gaussian kernel between [0,1]
g = Graphics()
x, mu, s = var('x, mu, s')
f(x,mu,s) = exp(-.5*((x-mu)/s)^2)
kernel_var = .02


for i in range(25):
    p = plot(f(x,.04*i + .02,kernel_var),(x,0,1), color=(.04*i, .5, .2))
    g += p
g.show()


### Plot the sum of all kernel's
def all_f(x):
    return sum([f(x,.04*i +.02,kernel_var) for i in range(25)])

plot(all_f(x), (x,0,1)).show()
