import numpy as np
import math
from PIL import Image
from scipy.linalg import hadamard
from scipy.ndimage.measurements import label

def cake_cutting(n):
	H=(hadamard(n) +np.ones(n))/2
	structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
	nComp =[]

	for i in range(n):
		basis=H[i,:]
		#non tutti sono quadrati!-> 2^5 non lo è
		#basis=basis.reshape((int(math.sqrt(n)),int(math.sqrt(n))))
		basis=basis.reshape(int(n/8),8)
		labeled, nC = label(basis, structure)
		nComp.append(nC)

	order = np.zeros((n,1)) -1
	min_prev = -1

	for i in range(n):
		prev = 0
		eq = 0
		for j in range(n):
			if nComp[j]<nComp[i]:
				prev +=1
			elif nComp[j] == nComp[i]:
				if i > j:
					eq+=1
		order[i] = prev + eq

	H1 = np.zeros((n,n))
	for i in range(n):
		
		H1[int(order[i]),:] = H[i, :]
	H1= 2*H1 -1
	return H1

cake_cutting(128)

