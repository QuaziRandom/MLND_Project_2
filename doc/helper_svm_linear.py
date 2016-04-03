import matplotlib.pyplot as pl
import numpy as np
from sklearn.svm import SVC

# Make a dummy dataset
X = np.array([[1, 2], [1.5, 3], [0.5, 4], [6, 2.5], [5, 4], [3, 5], [7, 7]])
y = np.array([0, 0, 0, 1, 1, 1, 1])

# Plot the dummy dataset
pl.figure(1)
pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired, s=100)
pl.title('Possible decision boundaries')
pl.xlabel('x1')
pl.ylabel('x2')

# Fit SVC on dummy data
clf = SVC(kernel='linear', C=.4, tol=1e-8, gamma=0.0001)
clf.fit(X, y)

# Show some possible separating lines
for intercept, slope in zip([-7,-25,3],[-2,-10,3]):
	x_dum = np.linspace(-1, 8)
	y_dum = slope * x_dum - intercept
	pl.plot(x_dum, y_dum, 'b-')

pl.tight_layout()

# Limit the graph
pl.xlim(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5)
pl.ylim(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5)

# Get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-1, 8)
yy = a * xx - (clf.intercept_[0]) / w[1]

# Get the parallels to the separating hyperplane that pass through the
# support vectors
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy + a * (margin + 0.5) # hard-code margin + 0.5; easier than tuning 
yy_up = yy - a * (margin + 0.5)   # params to get "hard" margins

# Plot the scatter again in a new figure
pl.figure(2)
pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired, s=100)
pl.title('Optimal decision boundary with margins')
pl.xlabel('x1')
pl.ylabel('x2')

# Plot the separating hyperplane and borders
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

# Limit the graph
pl.xlim(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5)
pl.ylim(np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5)

pl.tight_layout()

pl.show()
