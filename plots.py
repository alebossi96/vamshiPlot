from vamshiPlot import read_instruction as ri
from matplotlib.backends.backend_pdf import PdfPages
class Plots:
    def __init__(self, inst):
        self.inst = inst
        self.pdf = PdfPages(self.inst.output_name+".pdf")
        self.plot()   
