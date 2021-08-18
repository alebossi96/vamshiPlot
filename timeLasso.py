import numpy as np
from scipy.optimize import minimize

class timeLasso:
	def __init__(self, alpha : float):
		self.alpha = alpha
	def fit(self, M, Y):
		result = minimize(self.loss, beta_init, args=(X,Y),
		method='BFGS', options={'maxiter': 500})

		return
	def loss(self, M, Y, X) -> float: #xprev
		X_prev = X#dim mismatch
		sz = len(X)
		X_prev[1:] = X[:(sz-2)]
		return np.norm(Y-M*X)+ self.alpha * np.norm(X-X_prev,1)
