import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
from transformMatrix import transformMatrix
from hadamardOrdering import cake_cutting


def readSdt(fileName, numBanks):
	data = []
	for i in range(numBanks):
		zeros = int(np.log10(numBanks+1))
		if numBanks == 1:
			sdt = sdtfile.SdtFile(fileName+".sdt")
		elif i==0:       
		     sdt = sdtfile.SdtFile(fileName+"_c"+'0'*(zeros)+str(i)+".sdt")
		elif i<10:       
		     sdt = sdtfile.SdtFile(fileName+"_c"+'0'*(zeros)+str(i)+".sdt")
		elif i<100:       
		     sdt = sdtfile.SdtFile(fileName+"_c"+'0'*(zeros-1)+str(i)+".sdt")
		elif i<1000:       
		     sdt = sdtfile.SdtFile(fileName+"_c"+'0'*(zeros-2)+str(i)+".sdt")
		time = sdt.times[0][:]
		batch = sdt.data[0].shape[0];
		for j in range(batch):
			data.append(sdt.data[0][ j,:])
	return (time,data)
def plotData(time, data, name):
	plt.pcolormesh(data[1:,1400:1800])
	plt.xlabel("time gate [a.u.]")
	plt.ylabel("wavelenght [a.u.]")
	plt.savefig(name+"_v2.jpg")
	plt.close()
	plt.subplot(5,1,1)
	plt.plot(data[1:,1500])
	plt.subplot(5,1,2)
	plt.plot(data[1:,1550])
	plt.subplot(5,1,3)
	plt.plot(data[1:,1600])
	plt.ylabel("time gates")
	plt.subplot(5,1,4)
	plt.plot(data[1:,1650])
	plt.subplot(5,1,5)
	plt.plot(data[1:,1700])
	plt.xlabel("DMD line")
	plt.savefig(name+"tg.jpg")
	plt.close()
	plt.subplot(5,1,1)
	plt.plot(data[25,1400:1800])
	plt.subplot(5,1,2)
	plt.plot(data[26,1400:1800])
	plt.subplot(5,1,3)
	plt.plot(data[27,1400:1800])
	plt.ylabel("DMD lines")
	plt.subplot(5,1,4)
	plt.plot(data[28,1400:1800])
	plt.subplot(5,1,5)
	plt.plot(data[29,1400:1800])
	plt.xlabel("time gate [a.u]")
	
	plt.savefig(name+"lines.jpg")
def accumulate(data, nBasis):
	tot = np.zeros((nBasis,4096))
	i = 0
	for el in data:
		idx = i%nBasis		
		tot[idx,:]+= el
		i+=1
	return tot

#va considerato che sono 1 e -1

def reconstructHadamard(data, nBasis):
	dataNp = np.zeros((nBasis,4096)) #non nMeas perchè voglio dim uguali per la ricostruzione. Impongo però basi non misurate a 0
 
	i = 0
	for el in data:
		dataNp[i,:] = el 
		i+=1 
	H = 0.5*(cake_cutting(nBasis) + np.ones((nBasis, nBasis)))
	#data è 2D

	recons = np.zeros((nBasis,4096))   
	for i in range(4096):
		recons[:,i] = lsmr(H,dataNp[:,i])[0] #spero ordine corretto
	return recons

def convertToHad(data,nBasis):
	H = 0.5*(hadamard(nBasis) + np.ones((nBasis, nBasis)))
	#out = np.zeros((nBasis,4096))
	out = np.matmul(H,data)
	return out
		

"""
np.set_printoptions(threshold=np.inf)
H=cake_cutting(32) 
"""
		
nBanks =40
nBasis = 64
nMeas = 64
name = "0406/m9"
(time,data) = readSdt(name,nBanks);

tot = accumulate(data,nMeas)

#had = convertToHad(tot,nBasis)
tot =reconstructHadamard(tot,nBasis)

dim = tot.shape
for i in range(dim[0]):
	for j in range(dim[1]):	
		if(tot[i][j]<0):
			tot[i][j] = 0
res = sum(sum(tot))
print(res)		
plotData(time, tot,name)






