import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
from scipy.optimize import fsolve
from scipy.signal import peak_widths
from typing import List, Tuple
def readSdt(fileName: str) -> np.array:
	"""
	
	:param name fileName - filename
	"""
	sdt = sdtfile.SdtFile(fileName)
	data = np.zeros((len(sdt.times[0]),2))
	for i in range(len(sdt.times[0])):
		data[i][0]= sdt.times[0][i]
		data[i][1]= sdt.data[0][0][i]
	return data 

class wavelengthTroughTime:
	
	def timeRayleigh(self) -> Tuple[int,float]:
		
		data = readSdt(self.fileRay)
		Post0 = np.where(data[:,1] == np.amax(data[:,1]))[0][0]
		t0 = data[Post0,0]
		return (Post0,t0)

	def __init__(self,
			 filename: str,
			  fileRayleigh: str,
			   wl_0: float):
		self.filename = filename
		data = readSdt(filename)
		self.data = data[:,1]
		self.time = data[:,0]
		self.fileRay = fileRayleigh
		self.timeRay= self.timeRayleigh()[1]
		self.wl_0 = wl_0
		self.calibration()




	def timeToWl700(self,time: np.array) -> np.array: #from time to wavelength
		p = np.array([-1.4616e+26,    3.5911e+18,     -5.0642e+10 ,    9.6792e+02 ])
		return p[0]*time**3+p[1]*time**2+p[2]*time**1+p[3]
	def timeToWl900(self,time: np.array) -> np.array: #from time to wavelength
		p = np.array([ -3.3068e+27 ,  1.2054e+20,  -1.5128e+12,   7.4733e+03])
		return p[0]*time**3+p[1]*time**2+p[2]*time**1+p[3]
	def timeToWl700_2(self,time: np.array) -> np.array:
		#ATTENZIONE test0033 PER QUALCHE MOTIVO HA IL TGATE LA METÀ DI QUELLO CORRETTO FATTORE 2 NON SPIEGATO!
		p =np.array([  -4.6640e+26,   1.3148e+18,  -2.9028e+10,   8.2144e+02])
		return p[0]*time**3+p[1]*time**2+p[2]*time**1+p[3]
	def timeToWl(self, time: np.array) -> np.array:
		"""
		:param name time - filename
		"""
		if(self.wl_0>900):
			return self.timeToWl900(time)	
		return self.timeToWl700_2(time)
	def calibration(self) -> Tuple[np.array, np.array]:
		self.lmd = np.zeros(len(self.data))
		self.wavenumber = np.zeros(len(self.data))
		f = (lambda time : self.timeToWl(time) -self.wl_0)
		t_Ral = fsolve(f, x0 = self.wl_0 )[0]
		offset= t_Ral-self.timeRay
		for i in range(len(self.data)):
			dt =  self.time[i]+offset; # s
			self.lmd[i] = self.timeToWl(dt)
			self.wavenumber[i] = (1/self.wl_0-1/self.lmd[i])*1e7
		return (self.lmd, self.wavenumber)

	def fwhmResolution(self): #non è un tuple!
	#TODO scrivere meglio
		data = readSdt(self.fileRay)[:,1]
		pos0= self.timeRayleigh()[0]
		
		fwhm_idx = int(peak_widths(data, [pos0])[0])
		(lmd,wn) = self.calibration()
		fwhm_idx = int(fwhm_idx/2)
		fwhmTime = self.time[pos0-fwhm_idx]- self.time[pos0+fwhm_idx]
		fwhmWL= lmd[pos0-fwhm_idx]-lmd[pos0+fwhm_idx]
		fwhmWn = wn[pos0-fwhm_idx]-wn[pos0+fwhm_idx]
		return(fwhmTime,fwhmWL,fwhmWn)
	def positionWn(self, wn) -> int:
		for i in range(len(self.wavenumber)):
			if self.wavenumber[i]<wn:
				return i
		return len(self.wavenumber)
		
if __name__ == "__main__":
	meas = wavelengthTroughTime("2106/m4.sdt","2106/rayleigh.sdt",1000)


	(lmd,wn) = meas.calibration()
	plt.plot(wn[1700:2600], meas.data[1700:2600])
	plt.savefig("test.jpg")
	plt.close()
