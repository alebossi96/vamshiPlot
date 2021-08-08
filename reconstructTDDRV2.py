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
RAST = "rast"

HAD = "had"
nPoints = 4096
"""
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
class f_tT_:
	def __init__(self,fileName,nBanks,nBasis,nMeas, compress = True):
		self.nBanks =nBanks
		self.nBasis = nBasis
		self.nMeas = nMeas
		(self.time,self.data) = self.readSdt(fileName);
		
		self.compress = compress
		self.accumulate()
		
	def readSdt(self, fileName):
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
	def accumulate(self):
		self.tot = np.zeros((self.dim(),nPoints ))
		i = 0
		for el in self.data:
			if self.compress:
				idx = i%self.nBasis		
				self.tot[idx,:]+= el
			else:
				self.tot[i,:]= el
			i+=1
		return
		
		
	def getData(self):
		return self.tot
	#magari posso passare :
	def getIsCompress(self):
		return self.compress
	def getnBasis(self):
		return self.nBasis
	def getnMeas(self):
		return self.nMeas
	def dim(self):
		if self.compress:
			return self.nBasis
		return self.nMeas
	def getTime(self):
		return self.time
			
			
class reconstructTDDR:
	def __init__(self,data, rastOrHad, lambda_0 ,method="lsmr", Alpha=0, removeFirstLine = True):
	
		self.data = data
		self.rastOrHad = rastOrHad
		self.lambda_0 = lambda_0
		self.method = method
		self.removeFirstLine = removeFirstLine
		self.nBasis = self.data.getnBasis()
		if  rastOrHad != RAST and rastOrHad != HAD:
			sys.exit("It must be raster or HAD")
		if rastOrHad == RAST and not self.data.getIsCompress():
			sys.exit("It cannot be raster and not compressed!")
		if rastOrHad == RAST and self.removeFirstLine:
			sys.exit("It cannot be raster and remove first line!")
		
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
	def M(self):
		dim = self.data.getnBasis()
		H = 0.5*(hadamardOrdering.cake_cutting(dim) + np.ones((dim,dim)))
		if self.data.getIsCompress():
			return H
		matrix = np.zeros((self.sz, self.nBasis))
		for i in range(self.sz):
			matrix[i,:]= H[i%self.nBasis,:]
		return matrix
	def execute(self):

		if self.rastOrHad == HAD:
			self.reconstructHadamard()
		elif rastOrHad == RAST:
			self.recons = self.data.data()
		else:
			sys.exit("controlla se hai scritto RAST o HAD")
	def reconstructHadamard(self):

		self.reg.fit(self.M() ,self.data.getData())   
		self.recons = self.reg.coef_	
		
	def axis(self):
		self.calibrationToWl()
		self.calibrateToWn()
	def calibrationToWl(self):
		p = np.array([  1.51491966* 64/self.nBasis, 809.28844682])
		self.wavelength=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wavelength[i]=i*p[0]+p[1]
	def calibrateToWn(self):
		self.wn=np.zeros(self.nBasis)
		for i in range(self.nBasis):
			self.wn[i]=(1/self.lambda_0-1/self.wavelength[i])*1e7
			
	def wlAtTimeGate(self, tGate):
		idx = np.where(self.time == tGate)
		return self.recons[:,idx]
	def line(self, idx):
		return self.recons[idx,:]
	def wavenumber(self):
		if self.removeFirstLine:
			return self.wn[1:]
		return self.wn
	def time(self):
		return self.data.getTime()
	def reconstruction(self):
		self.recons = self.recons.T
		if self.rastOrHad == RAST:
			return self.recons
		if self.removeFirstLine:
			return self.recons[1:,:]
		return self.recons
		
	def fwhm(self,axis,data):
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

