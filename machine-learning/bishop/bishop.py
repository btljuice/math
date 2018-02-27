import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Generate some data for this demonstration.
mu = 10,
std = 2.5
data = norm.rvs(mu, std, size=500)

# # Fit a normal distribution to the data:
# mu, std = norm.fit(data)

# # Plot the histogram.
# plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()
