import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'red', 'purple', 'gold','green','darkorange'])
label_iter = itertools.cycle(['cluster'+str((i+1)) for i in range(100)])


def plot_results(X, Y, means, covariances, index, title):
    splt = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color,label) in enumerate(zip(means, covariances, color_iter,label_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], label=label, color=color, s=20, marker="o")
        # plt.xlabel('Figure1',fontsize = 'large')
        # plt.ylabel('Figure2',fontsize = 'large')
        plt.legend()

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splt.bbox)
        ell.set_alpha(0.5)
        splt.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    my_x_ticks = np.arange(-5, 5, 1)
    my_y_ticks = np.arange(-5, 5, 1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

# Parameters
n_samples = 1000

# Generate random sample following a sine curve
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(X.shape[0]):
    x = i * step - 6.
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))

plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.04, top=0.95, hspace=.2, wspace=.05,left=.03, right=.97)

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=6, covariance_type='full',max_iter=500).fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,'Expectation-maximization')

plt.show()
