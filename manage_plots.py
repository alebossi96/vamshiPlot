from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openpyxl
class BasePlot:
    def __init__(self, width, height, n_rows, n_cols, n_subplots = 1):
        self.width = width
        self.height = height
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_subplots = n_subplots
        self.gen_fig()
        self.adjust_plot()
    def adjust_plot(self):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                for k in range(self.n_subplots):
                    self.axs[i][j][k].grid(axis='both',which='major')
                    self.axs[i][j][k].tick_params(axis="both",which="major",direction="in",labelsize=4)
class MultiMultiplot(BasePlot):
    def gen_fig(self):
        self.fig = plt.figure(figsize=(self.width, self.height))
        if self.n_subplots>1:
            space = 0.1
        else:
            space = 0
        outer = gridspec.GridSpec(self.n_rows, self.n_cols, wspace=space, hspace=space)
        self.axs = []
        cont = 0
        for row in range(self.n_rows):
            col_axs = []
            for col in range(self.n_cols):
                sp_axs = []
                inner = gridspec.GridSpecFromSubplotSpec(self.n_subplots, 1,
                    subplot_spec=outer[cont], wspace=0, hspace=0)
                cont+=1
                for sp in range(self.n_subplots):
                    axs = plt.Subplot(self.fig, inner[sp])
                    self.fig.add_subplot(axs)
                    sp_axs.append(axs)
                col_axs.append(sp_axs)
            self.axs.append(col_axs)
    def set_x_labels(self, i, j, k, text):
        dims = self.axs[i][j][k].get_window_extent()
        pos = self.axs[i][j][k].get_position()
        self.fig.text((dims.width/2+pos.width)*1.1,(dims.height/2+pos.height)*1.1, text, va='center')

class SubplotNewAxis(BasePlot):
    def gen_fig(self):
        self.fig = plt.figure(figsize=(self.width, self.height))
        outer = gridspec.GridSpec(self.n_rows, self.n_cols, wspace=0, hspace=0)
        self.axs = []
        for i in range(self.n_rows):
            axs_col = []
            for j in range(self.n_cols):
                host = host_subplot(outer[i,j], axes_class=AA.Axes)
                axs_col.append([host, host.twinx()])
            self.axs.append(axs_col)
                
class NormalSubplot(BasePlot):
    def gen_fig(self):
        self.fig,axs = plt.subplots(self.n_rows,self.n_cols,sharex=True,sharey=True,figsize=(self.width,self.height),squeeze=False)
        plt.subplots_adjust(hspace=0,wspace=0,left=0.09,bottom=0.09)
        self.axs = []
        for i in range(self.n_rows):
            axs_col = []
            for j in range(self.n_cols):
                axs_col.append([axs[i,j]])
            self.axs.append(axs_col)
