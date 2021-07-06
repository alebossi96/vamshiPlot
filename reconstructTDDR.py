import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
import librerieTesi.hadamardOrdering as hadamardOrdering


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
	def __init__(self,fileName,nBanks,nBasis,nMeas, rastOrHad, lambda_0):
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
		self.tot = np.zeros((self.nBasis,self.nPoints ))
		i = 0
		for el in self.data:
			idx = i%self.nBasis		
			self.tot[idx,:]+= el
			i+=1
		return

	#va considerato che sono 1 e -1 le basi di Hadamard e io uso 1 e 0

	def reconstructHadamard(self):
		dataNp = np.zeros((self.nBasis,self.nPoints )) #non nMeas perchè voglio dim uguali per la ricostruzione. Impongo però basi non misurate a 0
	 
		i = 0
		for el in self.tot:
			dataNp[i,:] = el 
			i+=1 
		H = 0.5*(hadamardOrdering.cake_cutting(self.nBasis) + np.ones((self.nBasis, self.nBasis)))
		#data è 2D

		self.recons = np.zeros((nBasis,self.nPoints ))   
		for i in range(self.nPoints ):
			self.recons[:,i] = lsmr(H,dataNp[:,i])[0] #spero ordine corretto
		
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
			return self.recons[1:,:]
		return self.recons
"""
mancano calibrazioni
"""
if __name__=="__main__":
	print("test")	
	nBanks =40
	nBasis = 64
	nMeas = 64
	fileName = "0406/m9"
	lambda_0=780
	test = reconstructTDDR(fileName,nBanks,nBasis,nMeas, "Had", lambda_0)
	from latex import latex
	lat = latex("test")
	plot = []
	
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9" )
	plot.append(name)
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9" )
	plot.append(name)
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9" )
	plot.append(name)
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9" )
	plot.append(name)
	name = lat.plotData2D(test.time,test.wavenumber(),test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m9","m9" )
	plot.append(name)
	lat.newFigure(plot, "m9")


"""
devo ancora fare il plot nel caso in cui voglio stampare solo time gate oppure solo linee
"""




