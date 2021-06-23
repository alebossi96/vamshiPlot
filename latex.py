from wlTT import wavelengthTroughTime
import matplotlib.pyplot as plt
class latex:
	def __init__(self, filename):
		self.ext ="jpg"
		self.f = open(filename+".tex","w")
		self.f.write("\\documentclass{report}\n")
		self.f.write("\\usepackage{graphicx}\n")
		self.f.write("\\usepackage{subcaption}\n")
		self.f.write("\\usepackage[section]{placeins}")
		self.f.write("\\begin{document}\n")
	def __del__(self):
		self.f.write("\\end{document}\n")
		self.f.close()
		
	def plotData(self, x,y,x_label,x_um,plot_title,logOrlin,saveas ):
		plt.plot(x,y)
		if(logOrlin):
			plt.yscale('log')
		plt.xlabel(x_label+"["+x_um+"]")
		plt.ylabel("counts [a.u.]")
		plt.title(plot_title)
		plt.savefig(saveas+"."+self.ext)
		plt.close()
		return saveas
	def plotData2D(self, x,y,data,x_label,x_um,y_label,y_um,plot_title,saveas ):
		plt.rcParams['pcolor.shading'] ='nearest'
		plt.pcolormesh(x, y,data)
		plt.xlabel( x_label+" ["+x_um+"]")
		plt.ylabel(y_label+ "["+y_um+"]")
		plt.title(plot_title)
		plt.savefig(saveas+"."+self.ext)
		plt.close()
		return saveas
	def addSubFigure(self,name, newline):
		self.f.write("\\begin{subfigure}{.5\\textwidth}\n")
		self.includeGraphics(name)
		self.f.write("\end{subfigure}\n")
		if newline:
			self.f.write("\\vline\n")

	def includeGraphics(self, name):
		self.f.write("\\includegraphics[scale=0.4]{"+name+"."+self.ext+"}\n")
	def newSection(self,sectionName):
		self.f.write("\\section{"+sectionName+"}\n")
	
	def newFigure(self,toPlot,caption):
		self.f.write("\\begin{figure}[!htb]\n")
		if(len(toPlot)==1):	
			self.includeGraphics(toPlot[0])
		else:
			i = 0
			for plot in totPlot:
				i+=1
				self.addSubFigure(plot, i%2)
				
		self.f.write("\caption{"+caption+"}")
		self.f.write("\\end{figure}\n")	
		
if __name__ == "__main__":

	test = latex("test")

	folder = "2303/300m/"

	plot = []
	meas = wavelengthTroughTime(folder+"CaCO/CaCO_300Mono_3600sGain12.sdt",folder+"CaCO/Raleigh.sdt",785)
	name = test.plotData(meas.time,meas.data,"time of arrival", "ns","histogram",0, "test1" )
	plot.append(name)
	
	test.newFigure(plot, "Ray")

