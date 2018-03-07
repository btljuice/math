print(mean([1,2,3,4,5]))
print(std([1,2,2,4,5,6,8]))

### log-normal data generation
samples = [lognormvariate(2,3) for i in range(100)]
print('lognormal mean=', mean([log(i) for i in samples]))

### normal data generation and histogram display
dist = RealDistribution('gaussian',3)
samples = [dist.get_random_element()+2 for _ in range(1000)]
print('normal mean=', mean(samples))
T = stats.TimeSeries(samples)
show(T.plot_histogram(normalize=False,bins=30))

### binomial distribution
import scipy.stats
binom_dist = scipy.stats.binom(20,.05)
show(bar_chart([binom_dist.pmf(x) for x in range(21)]))

### Using R within sage
x=r([2.9, 3.0, 2.5, 2.6, 3.2]) # normal subjects
y=r([3.8, 2.7, 4.0, 2.4])      # with obstructive airway disease
z=r([2.8, 3.4, 3.7, 2.2, 2.0]) # with asbestosis
a = r([x,y,z]) # make a long R vector of all the data
b = r.factor(5*[1]+4*[2]+5*[3]) # create something for R to tell which subjects are which
a; b # show them
print(r.kruskal_test(a,b))

# Using %r. Does not work in console
%r
x = c(18,23,25,35,65,54,34,56,72,19,23,42,18,39,37) # ages of individuals
y = c(202,186,187,180,156,169,174,172,153,199,193,174,198,183,178) # maximum heart rate of each one
png() # turn on plotting
plot(x,y) # make a plot
lm(y ~ x) # do the linear regression
abline(lm(y ~ x)) # plot the regression line
dev.off()     # turn off the device so it plots
