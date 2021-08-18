
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
class latex:
	def __init__(self, filename: str):
		self.ext ="jpg"
		self.f = open(filename+".tex","w")
		self.f.write("\\documentclass{report}\n")
		self.f.write("\\usepackage{graphicx}\n")
		self.f.write("\\usepackage{subcaption}\n")
		self.f.write("\\usepackage[section]{placeins}\n")
		self.f.write("\\begin{document}\n")
	def __del__(self):
		self.f.write("\\end{document}\n")
		self.f.close()
		
	def plotData(self,
			 x: np.array,
			 y: np.array,
			 x_label: str,
			 x_um: str,
			 plot_title: str,
			 IsLog: bool,
			 saveas: str ) -> str:
		plt.plot(x,y)
		if(IsLog):
			plt.yscale('log')
		plt.xlabel(x_label+"["+x_um+"]")
		plt.ylabel("counts [a.u.]")
		plt.title(plot_title)
		plt.savefig(saveas+"."+self.ext)
		plt.close()
		return saveas
	def plotData2D(self,
			 x: np.array,
			 y: np.array,
			 data: np.array,
			 x_label: str,
			 x_um: str,
			 y_label: str,
			 y_um: str,
			 plot_title: str,
			 saveas: str ) -> str:
		plt.rcParams['pcolor.shading'] ='nearest'
		plt.pcolormesh(x, y,data)
		plt.xlabel( x_label+" ["+x_um+"]")
		plt.ylabel(y_label+ "["+y_um+"]")
		plt.title(plot_title)
		plt.colorbar()
		plt.savefig(saveas+"."+self.ext)
		plt.close()
		return saveas
	def addSubFigure(self,name, newline: bool) -> None:
		self.f.write("\\begin{subfigure}{.5\\textwidth}\n")
		self.includeGraphics(name)
		self.f.write("\end{subfigure}\n")
		if newline:
			self.f.write("\\vline\n")

	def includeGraphics(self, name) -> None:
		self.f.write("\\includegraphics[scale=0.4]{"+name+"."+self.ext+"}\n")
	def newSection(self,sectionName) -> None:
		self.f.write("\\section{"+sectionName+"}\n")
	def newSubSection(self,sectionName) -> None:
		self.f.write("\\subsection{"+sectionName+"}\n")
	
	def newFigure(self,toPlot: List[str],caption) -> None:
		self.f.write("\\begin{figure}[!h]\n")
		if(len(toPlot)==1):	
			self.includeGraphics(toPlot[0])
		else:
			i = 0
			for plot in toPlot:
				i+=1
				self.addSubFigure(plot, i%2)
				
		self.f.write("\n \caption{"+caption+"}\n\n")
		self.f.write("\\end{figure}\n")	
		
if __name__ == "__main__":
	
	test = latex("test")

	folder = "2303/300m/"
	from wlTT import wavelengthTroughTime
	plot = []
	meas = wavelengthTroughTime(folder+"CaCO/CaCO_300Mono_3600sGain12.sdt",folder+"CaCO/Raleigh.sdt",785)
	name = test.plotData(meas.time,meas.data,"time of arrival", "ns","histogram",0, "test1" )
	plot.append(name)
	
	test.newFigure(plot, "Ray")

