import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
from scipy.optimize import fsolve

def readSdt(fileName):
	sdt = sdtfile.SdtFile(fileName)
	data = np.zeros((len(sdt.times[0]),2))
	for i in range(len(sdt.times[0])):
		data[i][0]= sdt.times[0][i]
		data[i][1]= sdt.data[0][0][i]
	return data 

class wavelengthTroughTime:
	
	def timeRayleigh(self,filename):
		data = readSdt(filename)
		Post0 = np.where(data[:,1] == np.amax(data[:,1]))[0][0]
		t0 = data[Post0,0]
		return t0

	def __init__(self, filename, fileRayleigh, wl_0):
		self.filename = filename
		data = readSdt(filename)
		self.data = data[:,1]
		self.time = data[:,0]
		self.timeRay= self.timeRayleigh(fileRayleigh)
		self.wl_0 = wl_0




	def timeToWl(self,time): #from time to wavelength
		p = np.array([-1.4616e+26,    3.5911e+18,     -5.0642e+10 ,    9.6792e+02 ])
		return p[0]*time**3+p[1]*time**2+p[2]*time**1+p[3]
	

	def calibration(self):
		lmd = np.zeros(len(self.data))
		wavenumber = np.zeros(len(self.data))
		f = (lambda time : self.timeToWl(time) -self.wl_0)
		t_Ral = fsolve(f, x0 = self.wl_0 )[0]
		offset= t_Ral-self.timeRay
		for i in range(len(self.data)):
			dt =  self.time[i]+offset; # s
			lmd[i] = self.timeToWl(dt)
			wavenumber[i] = (1/self.wl_0-1/lmd[i])*1e7
		return(lmd, wavenumber)


meas = wavelengthTroughTime("2106/m4.sdt","2106/rayleigh.sdt",1000)


(lmd,wn) = meas.calibration()
plt.plot(wn[1700:2600], meas.data[1700:2600])
plt.savefig("test.jpg")
plt.close()
