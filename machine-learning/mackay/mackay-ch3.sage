#!/usr/bin/sagemath

### ex. 3.5

### Plot likelihood of a coin toss following a general Bernouilli distribution
p,a,b = var('p,a,b')
likelihood(p,a,b) = p^a*(1-p)^b
ll1 = plot(likelihood(p,2,1), (p,0,1), color='red', legend_label='a=1+1,b=0+1')
ll3 = plot(likelihood(p,6,1), (p,0,1), color='green', legend_label='a=5+1,b=0+1')
ll2 = plot(likelihood(p,3,3), (p,0,1), color='blue', legend_label='a=2+1,b=2+1')
show(ll1 + ll2 + ll3, axes_labels=['$p$', '$p^a(1-p)^b$'], title='likelihood')

### Plot posterior Distribution of coin toss, given prior is a uniform
pdf_prior = line([(p, RealDistribution('beta',[1,1]).distribution_function(p)) for p in srange(0,1, .01)],
                 color='red', legend_label='prior, a=1, b=1')
pdf_post21 = line([(p, RealDistribution('beta',[3,2]).distribution_function(p)) for p in srange(0,1, .01)],
                 color='blue', legend_label='posterior, s=aba')
pdf_post03 = line([(p, RealDistribution('beta',[1,4]).distribution_function(p)) for p in srange(0,1, .01)],
                  color='green', legend_label='posterior, s=bbb')
show(pdf_prior + pdf_post21 + pdf_post03, axes_labels=['$p$', '$pdf(p)$'], title='Beta Function')


### P.53 - log Evidence Ratio
p0 = 1/6
a,b,p = var('a,b,p')
# This is the log evidence ratio calculated manually. It equals the ln (Beta pdf^-1)
# We'll use it instead since it is faster
# Manual
# Pr_h0(a,b,p) = p^a*(1-p)^b
# Pr_h1(a,b) = factorial(a)*factorial(b)/factorial(a+b+1)
# log_ratio = lambda a,b: log(Pr_h1(a,b,p0)/Pr_h0(a,b,p0), 10)
# Beta pdf
log_ratio = lambda a,b: -log(RealDistribution('beta',[a+1,b+1]).distribution_function(p0), 10)
n = 40

### Contour plot of the log evidence ratio vs (fa, fb)
c = contour_plot(log_ratio, (0,n), (0,n),
                 cmap='terrain',colorbar=True , contours=[-3,-2,-1,0,1,2,3])
                 # cmap='terrain',colorbar=True , contours=128)

l5 = line([(x,5-x) for x in range(0,6)], color='red', legend_label='F=6')
l10 = line([(x,10-x) for x in range(0,11)], color='green', legend_label='F=10')
l20 = line([(x,20-x) for x in range(0,21)], color='orange', legend_label='F=20')
l40 = line([(x,40-x) for x in range(0,41)], color='purple', legend_label='F=40')
show(c+l5+l10+l20+l40, axes_labels=['$F_a$', '$F_b$'],
     title='Evi. log ratio: $\\log_{10}{\\frac{P[H_1|F_a,F_b]}{P[H_0|F_a,F_b]}}=-\\log_{10}{[pdf_{Beta}(F_a+1,F_b+1)]}$ contour plot')

### Show evidence log ratio
ev6  = plot(lambda a: log_ratio(a, 6-a), (0,6), color='red', legend_label='F=6')
ev10 = plot(lambda a: log_ratio(a, 10-a), (0,10), color='green', legend_label='F=10')
ev20 = plot(lambda a: log_ratio(a, 20-a), (0,20), color='orange', legend_label='F=20')
ev40 = plot(lambda a: log_ratio(a, 40-a), (0,40), color='purple', legend_label='F=40')
ev6.axes_labels_size(1)
ev10.axes_labels_size(1)
ev20.axes_labels_size(1)
ev40.axes_labels_size(1)

g = graphics_array(((ev6, ev10),(ev20,ev40)))
show(g, axes_labels=['$F_a$','Evi. log ratio=$\\log_{10}{\\frac{P[H_1|F_a,F_b]}{P[H_0|F_a,F_b]}}$'])


### P. 54 - Ex 3.7.
### Plot minimum and maximum value of the evidence ratio
# lr = Log Evidence ratio of H1 against H0 (H1/H0)
F = var('F')
p_h0 = 1/6  # H0 assumption
lr_max(F) = -(F*log(p_h0, 10) + log(F+1, 10))   # When Fa = F, Fb = 0
lr_min(F) = -(F*log(1-p_h0, 10) + log(F+1, 10)) # When Fb = F, Fa = 0

p = plot([lr_max(F), lr_min(F)], 0,20,
         color={0:'red', 1:'blue'},
         legend_label={0:'$F_a=F$',1:'$F_b=F$'},
         fill={0:[1]})
p.axes_labels_size(1)
p.show(axes_labels=['F', '$log_{10}(ratio)$'],
       title='Maximal Range of the evidence ratio')

### Plot confidence intervals
from scipy.stats import binom
from bisect import bisect_right
from random import random
import numpy as np

def log_ratio_triplets(F, pr_a):
    return [(fa, log_ratio(fa,F-fa), binom.pmf(fa, F, pr_a)) for fa in range(0,F+1)]


def log_ratio_probs(F, pr_a):
    triplets = log_ratio_triplets(F, pr_a)
    probs = [ t[2] for t in triplets]
    values = [ t[1] for t in triplets]
    return (values, probs)


def discrete_unimodal_confidence_interval(values, weights, ci_percentage):
    sorted_probs_and_values = sorted(zip(weights/sum(weights), values),reverse=True)

    cum_probs = np.cumsum([ x[0] for x in sorted_probs_and_values ])
    values = [ x[1] for x in sorted_probs_and_values ]

    i = max(1, bisect_right(cum_probs, ci_percentage))
    return (min(values[:i]), max(values[:i]))


def log_ratio_confidence_interval(F, pa_theorical, ci_percentage):
    ci = []
    for f in range(1,F+1):
        values, weights = log_ratio_probs(f, pa_theorical)
        r_min, r_max = discrete_unimodal_confidence_interval(values, weights, ci_percentage)
        ci.append((f, r_min, r_max))
    return ci


def log_ratio_confidence_interval_plot(F, pa_theorical, ci_percentage, color):
    ci = log_ratio_confidence_interval(F, pa_theorical, ci_percentage)
    ci_lower = line([(x[0], x[1]) for x in ci],
                    color=color)
                    # legend_label='%.2f ci' % ci_percentage)
    ci_upper = line([(x[0], x[2]) for x in ci],
                    color=color)
    return ci_lower + ci_upper


def log_ratio_confidence_interval_plots(F, pa_theorical):
    ci1= log_ratio_confidence_interval_plot(F, pa_theorical, .95, 'red')
    ci2 = log_ratio_confidence_interval_plot(F, pa_theorical, .80, 'green')
    ci3 = log_ratio_confidence_interval_plot(F, pa_theorical, .50, 'blue')
    ci = ci1 + ci2 + ci3
    ci.axes_labels_size(1)
    ci.legend(True)
    ci.set_legend_options(title='pa = %.2f'%pa_theorical)
    return ci


# Try to plot confidence intervals
g = graphics_array(((log_ratio_confidence_interval_plots(100, 1/6),
                     log_ratio_confidence_interval_plots(100, .25)),
                    (log_ratio_confidence_interval_plots(100, .33),
                     log_ratio_confidence_interval_plots(100, .5))))

show(g, figsize=[8,8], axes_labels=['$F_a$','Evi. log ratio'])


### Empirical sampling
def sample_log_ratio(F, pa_theorical, n_experience=100):
    samples = np.random.binomial(1, pa_theorical, (n_experience, F))
    samples = np.cumsum(samples,axis=1)
    ratios = np.ndarray((n_experience, F))
    for e in range(n_experience):
        for f in range(F):
            fa = samples[e,f]
            ratios[e,f]  = log_ratio(fa, f+1-fa)
    return (samples, ratios)


def sample_log_ratio_plot(F, pa_theorical, n_experience=100, n_plots=3):
    _unused, ratios = sample_log_ratio(F, pa_theorical, n_experience)

    # Plot first samples
    p = None
    for i in range(n_plots):
        points = zip(range(1,F+1), ratios[i])
        if p is None:
            p = line(points)
        else:
            p = p + line(points)

    # Plot mean and ic
    mu = np.mean(ratios, axis=0)
    sigma =  np.std(ratios, axis=0,ddof=1)
    p += line(zip(range(1,F+1), mu), color='green' )
    p += line(zip(range(1,F+1), mu-sigma), color='orange' )
    p += line(zip(range(1,F+1), mu+sigma), color='orange' )
    p += line(zip(range(1,F+1), mu-3*sigma), color='red' )
    p += line(zip(range(1,F+1), mu+3*sigma), color='red' )

    return p

g = graphics_array(((sample_log_ratio_plot(100, 1/6, 1000),
                     sample_log_ratio_plot(100, .25, 1000)),
                    (sample_log_ratio_plot(100, .33, 1000),
                     sample_log_ratio_plot(100, .5, 1000))))

show(g, figsize=[8,8], axes_labels=['$F_a$','Evi. log ratio'])


