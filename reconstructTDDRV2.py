import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
import librerieTesi.hadamardOrdering as hadamardOrdering
from sklearn import linear_model
from scipy.signal import peak_widths
import sys
import copy
from typing import List, Tuple
RAST = "rast"
#assert False, "Oh no! This assertion failed!"
HAD = "had"
nPoints = 4096
"""
USARE scipy.optimize
class ridgeTime:
	def __init__(self,alpha):
		self.alpha = alpha
	def fit(self, matrix, data):
		self.recons = np.zeros((nBasis,self.nPoints ))
		lmd = 10;
		prev = np.zeros(self.recons[:,0].shape)
		for i in range(self.nPoints ):
			M = np.matmul(self.matrix.T,self.matrix)
			M += lmd *np.eye(M.shape[0])
			b = np.matmul(self.matrix.T, dataNp[:,i]) +lmd* prev
			self.recons[:,i] = lsmr(M,b)[0]
			prev = self.recons[:,i]
	
"""

#TODO data type f_tT_
#TODO mypy
class f_tT_:
	def __init__(
			self,
			fileName: str,
			nBanks: int,
			nBasis: int,
			nMeas: int = -1,
			compress: bool = True):
		self.nBanks =nBanks
		self.nBasis = nBasis
		if nMeas == -1: # se non specificato è uguale a nBasis, (caso compressioni)
			self.nMeas = nBasis
		else:
			self.nMeas = nMeas
		(self.time,self.data) = self.readSdt(fileName);
		
		self.compress = compress
		self.accumulate()
		
	def readSdt(self, fileName: str) -> Tuple[np.array,List]:
		data = []
		for i in range(self.nBanks):
			zeros = int(np.log10(self.nBanks+1))
			fn = fileName
			if self.nBanks == 1:
				sdt = sdtfile.SdtFile(fn+".sdt")
			elif i==0:       
			     sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
			elif i<10:       
			     sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
			elif i<100:       
			     sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-1)+str(i)+".sdt")
			elif i<1000:       
			     sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-2)+str(i)+".sdt")
			elif i<10000:       
			     sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-3)+str(i)+".sdt")
			time = sdt.times[0][:]
			batch = sdt.data[0].shape[0];
			for j in range(batch):
				data.append(sdt.data[0][ j,:])
		return (time,data)
	def accumulate(self) -> None:
		self.tot = np.zeros((self.dim(),nPoints ))
		
		i = 0
		for el in self.data:
			if self.compress:
				idx = i%self.nBasis		
				self.tot[idx,:]+= el
			else:
				self.tot[i,:]= el
			i+=1
		
	def getData(self) -> np.array:
		return self.tot
	#magari posso passare :
	def getIsCompress(self) -> bool:
		return self.compress
	def getnBasis(self) -> int:
		return self.nBasis
	def getnMeas(self) -> int:
		return self.nMeas
	def dim(self) -> int:
		if self.compress:
			return self.nBasis
		return len(self.data)#ho copiato tutto, cambio solo tot
	def getTime(self) -> np.array:
		return self.time
	def copy(self,i0: int, iF: int):
		new = copy.copy(self)
		new.nMeas = iF-i0
		new.data = self.data[i0:iF]
		new.tot =self.tot[i0:iF,:]
		return new
	def __add__(self,b):
		return self.tot+b.tot
	def __sub__(self,b):
		return self.tot-b.tot
class reconstructTDDR:
	def __init__(self,
			data: f_tT_, 
			rastOrHad: str, 
			lambda_0: int ,
			method: str="lsmr", 
			Alpha: float=0, 
			removeFirstLine: bool = True):
	
		self.data = data
		self.rastOrHad = rastOrHad
		self.lambda_0 = lambda_0
		self.method = method
		if rastOrHad == RAST:
			self.removeFirstLine = False
		else:
			self.removeFirstLine = removeFirstLine
		self.nBasis = self.data.getnBasis()
		assert not(rastOrHad != RAST and rastOrHad != HAD), "It must be RAST or HAD"

		assert not (rastOrHad == RAST and not self.data.getIsCompress()),"It cannot be raster and not compressed!"
		assert not(rastOrHad == RAST and self.removeFirstLine), "It cannot be raster and remove first line!"
		
		if method == "lsmr":
			print("lsmr")
			self.reg = linear_model.LinearRegression()
		elif method == "Lasso":
			print("Lasso")
			self.reg = linear_model.Lasso(alpha=Alpha)
		elif method == "Ridge":
			print("Ridge")
			self.reg = linear_model.Ridge(alpha=Alpha)
			
		self.sz = self.data.dim()
		self.execute()	
		self.axis()
	def M(self) -> np.array:
		dim = self.data.getnBasis()
		H = 0.5*(hadamardOrdering.cake_cutting(dim) + np.ones((dim,dim)))
		if self.data.getIsCompress():
			return H
		matrix = np.zeros((self.sz, self.nBasis))
		
		for i in range(self.sz):
			matrix[i,:]= H[i%self.nBasis,:]
		return matrix
	def execute(self) -> None:

		if self.rastOrHad == HAD:
			self.reconstructHadamard()
		elif self.rastOrHad == RAST:
			self.recons = self.data.getData()
		else:
			sys.exit("controlla se hai scritto RAST o HAD")
	def reconstructHadamard(self) -> None:
		"""
		self.reg.fit(self.M() ,self.data.getData())  
		self.reg.fit(self.M() ,self.data.getData())   
		self.recons = self.reg.coef_	 
		print(self.M().shape)
		print(self.data.getData().shape)
		

		"""
		self.reg.fit(self.M() ,self.data.getData())   
		recons1 = self.reg.coef_
		"""	
		recons2 = np.zeros((self.nBasis,4096)) 
		for i in range(4096):
			recons2[:,i] = lsmr(self.M(),self.data.getData()[:,i])[0] #Perchè qui c'è la prima linea a caso e in reg..fit no?
		"""
		self.recons= recons1#self.recons.T

	def axis(self) -> None:
		self.calibrationToWl()
		self.calibrateToWn()
	def calibrationToWl(self) -> None:
		p = np.array([  1.51491966* 64/self.nBasis, 809.28844682])
		self.wl=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wl[i]=i*p[0]+p[1]
	def calibrateToWn(self) -> None:
		self.wn=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wn[i]=(1/self.lambda_0-1/self.wl[i])*1e7
			
	def wlAtTimeGate(self, tGate: int) -> np.array:
		idx = np.where(self.time == tGate)
		return self.recons[:,idx]
	def line(self, idx: int) -> np.array:
		return self.recons[idx,:]
	def wavenumber(self) -> np.array:
		if self.removeFirstLine:
			return self.wn[1:]
		return self.wn
	def wavelength(self) -> np.array:
		if self.removeFirstLine:
			return self.wl[1:]
		return self.wl
	def time(self, inNs: bool = True) -> np.array:
		if inNs:
			conv = 1e9
		else:
			conv = 1
		return self.data.getTime() * conv
	def reconstruction(self) -> np.array:
		if self.rastOrHad == RAST:
			return self.recons
		recons = self.recons.T
		if self.removeFirstLine:
			return recons[1:,:]
		return recons
		
	def fwhm(self,axis: np.array,data: np.array) -> float:
		pos0 = np.where(data == np.amax(data))[0][0]
		fwhm_idx = int(peak_widths(data, [pos0])[0])
		
		fwhm =  (axis[pos0+fwhm_idx] - axis[pos0-fwhm_idx])*1000
		return fwhm
		
if __name__=="__main__":
	
	nBanks =20
	nBasis = 32
	nMeas = 32
	fileName = "0406/m1"
	lambda_0=780
	data = f_tT_(fileName,nBanks,nBasis,nMeas, compress = True)
	test = reconstructTDDR(data, HAD, lambda_0,method = "Ridge", Alpha= 100, removeFirstLine =False)
	from latex import latex
	lat = latex("test")
	plot = []

	name = lat.plotData2D(test.time() , test.wavenumber(), test.reconstruction(),"time","ns","wavenumber","$cm^{-1}$","reconstruct m1","m1RidgeoComp" )

