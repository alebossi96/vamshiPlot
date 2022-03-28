from VamshiPlot import read_instruction as ri
from matplotlib.backends.backend_pdf import PdfPages
class Plots:
    def __init__(self, inst):
        self.pdf = PdfPages(inst.output_name+".pdf")
        self.plot()   
