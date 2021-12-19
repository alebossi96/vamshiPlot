import numpy as np
from scipy.optimize import minimize
#TODO problema dimensioni matrici
#TODO controlla con grad descent
class timeLasso:
	def __init__(self, alpha : float):
		self.alpha = alpha
		
		
	def fit(self, M, Y):
	
		self.nBasis = M.shape[1]
		self.dim = M.shape[0]
		self.nPoints = Y.shape[1]
		Y = self.toVec(Y)
		X_0 = np.zeros( (self.nBasis, self.nPoints,)) #self.toVec(X_0)
		self.M = M
		coef_ = minimize(self.loss, X_0, args=(Y),
		method='Newton-CG', options={'maxiter': 500})

		return
	def error(self, X, Y) -> float: #xprev
		X_prev = X#dim mismatch
		sz = len(X)
		M = self.M
		X_prev = np.roll( X, 2, axis =1)
		#X_prev[0] =  0

		return np.linalg.norm(Y-M.dot(X))+ self.alpha * np.linalg.norm(X-X_prev,1)
	def toVec(self, X):
		return X.flatten()
	def toMat(self, Y, shape): #che cosa sto aggiungendo?
		
		res = np.zeros(shape)
		for i in range(shape[0]):
			I0 = i*shape[0]
			If = (i+1)*shape[0] 
			res[:,i] = Y[I0:If]
		return res
	def loss(self, X, Y):
		X = self.toMat(X, (self.nBasis, self.nPoints))
		
		Y = self.toMat(Y, (self.dim, self.nPoints))

		return self.error(X,Y)
	
	
if __name__=="__main__":
	
	t = timeLasso(0.1)
	M = np.array([[0,1],[1,0]])
	X_0 = np.array([[2,2],[0,0]])
	Y = np.array([[0,0],[0,0]])
	t.fit(M,X_0,Y)
