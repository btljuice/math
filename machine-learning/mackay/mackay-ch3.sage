#!/usr/bin/sagemath

### ex. 3.5
p,a,b = var('p,a,b')

### Plot likelihood
likelihood(p,a,b) = p^a*(1-p)^b
ll1 = plot(likelihood(p,2,1), (p,0,1), color='red', legend_label='a=2,b=1')
ll3 = plot(likelihood(p,6,1), (p,0,1), color='green', legend_label='a=6,b=1')
ll2 = plot(likelihood(p,3,3), (p,0,1), color='blue', legend_label='a=3,b=3')
show(ll1 + ll2 + ll3, axes_labels=['$p$', '$p^a(1-p)^b$'], title='likelihood')

### Plot posterior
pdf_prior = line([(p, RealDistribution('beta',[1,1]).distribution_function(p)) for p in srange(0,1, .01)],
                 color='red', legend_label='prior, a=1, b=1')
pdf_post21 = line([(p, RealDistribution('beta',[3,2]).distribution_function(p)) for p in srange(0,1, .01)],
                 color='blue', legend_label='posterior, s=aba')
pdf_post03 = line([(p, RealDistribution('beta',[1,4]).distribution_function(p)) for p in srange(0,1, .01)],
                  color='green', legend_label='posterior, s=bbb')
show(pdf_prior + pdf_post21 + pdf_post03, axes_labels=['$p$', '$pdf(p)$'], title='Beta Function')


### P.53 - Evidence Ratio
p0 = 1/6
# This is the evidence ratio calculated manually. It equals the Beta pdf.
# We'll use it instead since it is faster
# Manual
# Pr_h0(a,b,p) = p^a*(1-p)^b
# Pr_h1(a,b) = factorial(a)*factorial(b)/factorial(a+b+1)
# ratio = lambda a,b: Pr_h0(a,b,p0)/Pr_h1(a,b)
# Beta pdf
ratio = lambda a,b: RealDistribution('beta',[a+1,b+1]).distribution_function(p0)
n = 40
c = contour_plot(ratio, (0,n), (0,n),
                 cmap='terrain',colorbar=True, contours=[.25,.5,1,2])
l5 = line([(x,5-x) for x in range(0,6)], color='red', legend_label='F=6')
l10 = line([(x,10-x) for x in range(0,11)], color='green', legend_label='F=10')
l20 = line([(x,20-x) for x in range(0,21)], color='orange', legend_label='F=20')
l40 = line([(x,40-x) for x in range(0,41)], color='purple', legend_label='F=40')
show(c+l5+l10+l20+l40, axes_labels=['$F_a$', '$F_b$'],
     title='Evidence ratio $\\frac{P[H_0|F_a,F_b]}{P[H_1|F_a,F_b]}=BetaDist(F_a+1,F_b+1)$ contour plot')
# contours = c+l5+l10+l20

ev6  = plot(lambda a: ratio(a, 6-a), (0,6), color='red', legend_label='F=6')
ev10 = plot(lambda a: ratio(a, 10-a), (0,10), color='green', legend_label='F=10')
ev20 = plot(lambda a: ratio(a, 20-a), (0,20), color='orange', legend_label='F=20')
ev40 = plot(lambda a: ratio(a, 40-a), (0,40), color='purple', legend_label='F=40')
ev6.axes_labels_size(1)
ev10.axes_labels_size(1)
ev20.axes_labels_size(1)
ev40.axes_labels_size(1)

g = graphics_array(((ev6, ev10),(ev20,ev40)))
show(g, axes_labels=['$F_a$','Evi. ratio=$\\frac{P[H_0|F_a,F_b]}{P[H_1|F_a,F_b]}$'])
