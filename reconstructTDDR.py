import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
import librerieTesi.hadamardOrdering as hadamardOrdering
from sklearn import linear_model

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

class reconstructTDDR:
	def __init__(self,fileName,nBanks,nBasis,nMeas, rastOrHad, lambda_0,method, lassoAlpha=0, compress = True):
		self.nBanks =nBanks
		self.nBasis = nBasis
		self.nMeas = nMeas
		(self.time,self.data) = readSdt(fileName,self.nBanks);
		self.rastOrHad=rastOrHad
		self.nPoints = 4096
		self.rmv = False
		self.lambda_0 = lambda_0
		self.calibrationToWl()
		self.calibrateToWn()
		self.compress = compress
		self.method=method
		if method == "lsmr":
			print("lsmr")
			self.reg = linear_model.LinearRegression()
		elif method == "Lasso":
			print("Lasso")
			self.reg = linear_model.Lasso(alpha=lassoAlpha)
		elif method == "Ridge":
			print("Ridge")
			self.reg = linear_model.Ridge(alpha=lassoAlpha)
				
		self.execute()

	def calibrationToWl(self):
		p = np.array([  1.51491966* 64/self.nBasis, 809.28844682])
		self.wavelength=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wavelength[i]=i*p[0]+p[1]
	def calibrateToWn(self):
		self.wn=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wn[i]=(1/self.lambda_0-1/self.wavelength[i])*1e7
	def execute(self):
		self.accumulate()
		if self.rastOrHad == "Had":
			self.reconstructHadamard()
			
			
	def accumulate(self):
		H =  0.5*(hadamardOrdering.cake_cutting(self.nBasis) + np.ones((self.nBasis, self.nBasis)))
		if self.compress:
			self.tot = np.zeros((self.nBasis,self.nPoints ))
			i = 0
			for el in self.data:
				idx = i%self.nBasis		
				self.tot[idx,:]+= el
				i+=1
			self.matrix = H
			return
		sz = len(self.data)
		self.tot = self.data
		self.matrix = np.zeros((sz, self.nBasis))
		for i in range(sz):
			self.matrix[i,:]= H[i%self.nBasis,:]
			
	#va considerato che sono 1 e -1 le basi di Hadamard e io uso 1 e 0

	def reconstructHadamard(self):
		sz = len(self.tot)
		dataNp = np.zeros((sz,self.nPoints )) #non nMeas perchè voglio dim uguali per la ricostruzione. Impongo però basi non misurate a 0
	 
		i = 0
		for el in self.tot:
			dataNp[i,:] = el 
			i+=1 
		
		#data è 2D

		#self.recons = np.zeros((nBasis,self.nPoints ))
		"""
		self.reg.fit(self.matrix,dataNp)   
		self.recons = self.reg.coef_
		print(self.recons.shape)
		"""
		for i in range(self.nPoints ):
			self.reg.fit(self.matrix,dataNp[:,i])
			self.recons[:,i] = self.reg.coef_
		
		#"""
	def wlAtTimeGate(self, tGate):
		idx = np.where(self.time == tGate)
		return self.recons[:,idx]
	def line(self, idx):
		return self.recons[idx,:]
	def removeFirstLine(self, rmv):
		self.rmv = rmv
	def wavenumber(self):
		if self.rmv:
			return self.wn[1:]
		return self.wn
	def reconstruction(self):
		if self.rmv:

			return self.recons[:,1:]
		return self.recons
"""
mancano calibrazioni
"""
if __name__=="__main__":
	
	nBanks =40
	nBasis = 64
	nMeas = 64
	fileName = "0406/m9"
	lambda_0=780
	test = reconstructTDDR(fileName,nBanks,nBasis,nMeas, "Had", lambda_0,"Ridge", 100, True)
	test.removeFirstLine(True)
	from latex import latex
	lat = latex("test")
	plot = []
	print(test.reconstruction().shape)
	name = lat.plotData2D(test.wavenumber(), test.time,test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9RidgeoComp" )
	test = reconstructTDDR(fileName,nBanks,nBasis,nMeas, "Had", lambda_0,"Lasso", 0.1, True)
	test.removeFirstLine(True)
	from latex import latex
	lat = latex("test")
	plot = []
	print(test.reconstruction().shape)
	name = lat.plotData2D(test.wavenumber(), test.time,test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9LassoComp" )
	test = reconstructTDDR(fileName,nBanks,nBasis,nMeas, "Had", lambda_0,"lsmr", 0.02, True)
	test.removeFirstLine(True)
	from latex import latex
	lat = latex("test")
	plot = []
	print(test.reconstruction().shape)
	name = lat.plotData2D(test.wavenumber(), test.time,test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9LsmrComp" )

	"""
	test = reconstructTDDR(fileName,nBanks,nBasis,nMeas, "Had", lambda_0,"lsmr", 0, False)
	test.removeFirstLine(True)
	from latex import latex
	lat = latex("test")
	plot = []
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9Comp" )
	"""

"""
devo ancora fare il plot nel caso in cui voglio stampare solo time gate oppure solo linee
"""




