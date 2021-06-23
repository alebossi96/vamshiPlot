import numpy as np
from scipy.linalg import hadamard
tol=1e-10



#def transformMatrix(nBasis,nPixelDMD):
def transformMatrix(arrivo,partenza):
	n=arrivo
	m=partenza
	if(n>m):
		return transformMatrix_N_greater_M(n,m)
	elif(n<m):
		return transformMatrix_N_greater_M(m,n).T
	else:
		return np.eye(n)

def transformMatrix_N_greater_M(n,m):
	a = np.zeros((n,m))
	a[0][0]=1
	col_sum =np.zeros(m)
	row_sum = np.zeros(n)
	col_sum[0]=1
	row_sum[0]=1
	for i in range(1,n):
		for j in range(0,m):

			col_value_max=max(n/m-col_sum[j],0)

			row_value_max = max(1-row_sum[i],0)
			a[i][j] = min(col_value_max,row_value_max)

			col_sum[j]+=a[i][j]

			row_sum[i]+=a[i][j]
			if(abs(col_sum[j]-n/m)<tol):
				continue
		if(abs(row_sum[i]-1)<tol):
			continue

			
	return a
'''

#aggiungere le funzioni necessarie che mi servono dopo per la ricustruzione
n=2
m=3

t_n_m=transformMatrix(n,m)
print(t_n_m)
H=hadamard(n)
print(H)

k=1
t_k_n=transformMatrix(k,n)
print(t_k_n)

H_DMD = np.matmul(H, t) #basi che dovranno essere usate nel DMD

print(H_DMD)

#ora voglio ricostruire base di partenza 
s1 = np.matmul(Transf,t.T)
TTT= np.matmul(t,t.T)
if(n>m):#immagino sia lento
	TTT_i=np.linalg.pinv(TTT)
else:
	TTT_i=np.linalg.inv(TTT)	

recons = np.matmul(s1, TTT_i)
print(recons)
'''
