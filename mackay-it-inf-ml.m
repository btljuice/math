# ex. 2.8

n = [ 0 2  3  29 ]
N = [ 3 3 10 300 ]
a = n + 1
b = N - n + 1
lh = n ./ N  # Likelihood estimate
u = a ./ (a + b)  # mean of the posterior

# Plot first 3 answers
# Observation:
# - The maximum likelihood method estimate of the probability of tossing a head
#   at the N+1th outcome corresponds to the "mode" of the bayesian posterior in this case.
# - The bayesian method estimate of the probability of tossing a head
#   at the N+1th outcome corresponds to the "mean" of the bayesian posterior in this case
x = ( 0:100 ) * .01
g = legend(plot(x, betapdf(x,a(1),b(1)), "r;a=1,b=4;",
            x, betapdf(x,a(2),b(2)), "g;a=3,b=2;",
            x, betapdf(x,a(3),b(3)), "b;a=4,b=8;",
            u(1), betapdf(u(1),a(1),b(1)), "or",
            u(2), betapdf(u(2),a(2),b(2)), "og",
            u(3), betapdf(u(3),a(3),b(3)), "ob",
            lh(1), betapdf(lh(1),a(1),b(1)), "*r",
            lh(2), betapdf(lh(2),a(2),b(2)), "*g",
            lh(3), betapdf(lh(3),a(3),b(3)), "*b"))
saveas(g, "mackay-ch2-ex2_8.pdf")

# Information content function = -x * log(x)
function ret = f(x)
  ret = -x.*log2(x)
endfunction
function ret = df(x)
  ret = -1/log(2).*(log(x) + 1)
endfunction
x = (1:100)*.01
g = plot(x, f(x), "r;-x*log2(x);",
     x, df(x), "g;derivative;",
     exp(-1), f(exp(-1)), "*b")
ylim([-1 2])
grid()
saveas(g, 'mackay-ch2-information-content-xlnx.pdf', 'pdf')

# ex 2.16 b)
n = 1:6
u = 100*mean(n)
s = (100*var(n,1))^.5
x = (1:8*s) + u - 4*s
y = normpdf(x, u, s)
plot(x,y)

# ex. 2.17
function ret = p(a)
  ret = 1./(1 + exp(-a))
endfunction
x = -10:.1:10
plot(x, p(x),
     x, tanh(x/2)./2+1/2)
grid()

# ex 2.20
function ret = f(n, e, r)
  ret = 1 - (1 - e./r).^n
endfunction
n = [ 2 2 10 10 1000 1000]
e = [ 1 1  1  1    1    1]
r = [ 100 2 100 2 100 2 ]
f(n,e,r)

# ex 2.34
x = 0:10
f_est = 1./(1+x)
y = nbinpdf(x, 1, 1/2)

# ex 2.39
n = 1:12367
p = .1./n
sum(-p.*log2(p))



