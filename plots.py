import math
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from vamshiPlot import read_instruction as ri
from vamshiPlot import manage_plots as mp
from vamshiPlot import core
import pandas as pd
#TODO aggiungere host_subplot & fare tutto con GridSpec da(https://stackoverflow.com/questions/43444108/how-can-i-create-different-sized-subplots-when-using-host-subplot)
              
class Plots:
    def __init__(self, instr):
        self.instr = instr
        self.pdf = PdfPages(self.instr.output_name+".pdf")
        self.width = instr.width
        self.height = instr.height
        self.context = core.Context()
        self.plot()
        
    def plot(self):
        for self.context.idx in range(self.instr.length):
            pages = [" "]
            if self.instr.page is None or self.instr.page.isnull()[self.context.idx]:
                self.context.skip_pages = True
            else:
                self.context.skip_pages = False
                pages = self.remove_duplicate_order_list(self.instr.data[self.instr.page[self.context.idx]])
            for self.context.page in pages:
                self.data = self.instr.select_data(self.context.idx)#errore
                if pd.isnull(self.instr.rows[self.context.idx]):
                    single_el = len(set(self.data[self.instr.columns[self.context.idx]]))
                    n_rows = int(math.sqrt(single_el))
                    n_cols = math.ceil(single_el/n_rows)
                else:
                    n_rows = len(set(data[self.instr.rows[self.context.idx]]))
                    n_cols = len(set(data[self.instr.columns[self.context.idx]]))
                self.mp = mp.MultiMultiplot(width= self.instr.width, height = self.instr.height, n_rows = n_rows, n_cols = n_cols, n_subplots = self.instr.num_subplots)
                i_col = 0
                i_row = 0
                if not self.context.skip_pages:
                    data = self.data[(self.data[self.instr.page[self.context.idx]]==self.context.page)]
                for self.context.col,self.context.c in enumerate(self.remove_duplicate_order_list(self.data[self.instr.columns[self.context.idx]])):
                    data = self.data[(self.data[self.instr.columns[self.context.idx]]==self.context.c)]
                    xlabel = "%s"%self.instr.columns[self.context.idx]+" \n= " +"%s"%self.context.c
                    if pd.isnull(self.instr.rows[self.context.idx]):
                        self.plot_in_grid(data_pivot =data,
                                           axs = self.mp.axs[i_row][i_col],
                                           tmp_label = "%s"%self.context.c)
                        ylabel = xlabel

                    else:
                        for self.context.row,self.context.r in enumerate(self.remove_duplicate_order_list(data[self.instr.columns[self.context.idx]])):
                            self.plot_in_grid(data_pivot = data[(data[self.instr.rows[self.context.idx]]==self.context.r)],
                                                    axs = self.mp.axs[i_row][i_col],
                                            tmp_label = "%s"%self.context.r+", %s"%self.context.c)
                            ylabel = "%s"%self.instr.rows[self.context.idx]+" \n= " +"%s"%self.context.r
                    if i_col == 0:
                        self.mp.axs[i_row][0][0].set_ylabel(ylabel)
                    if i_row == 0:
                        self.mp.axs[0][i_col][0].set_title(xlabel)
                    i_col+=1
                    if i_col == n_cols:
                        i_col = 0
                        i_row+=1
                if self.context.skip_pages:
                    self.set_labels()
                else:
                    self.set_labels(page = self.context.page)
                plt.show()
                self.pdf.savefig(self.mp.fig)
                plt.clf()
        self.pdf.close()
    def plot_in_grid(self, data_pivot, axs, tmp_label = None):
        for self.context.i, y_subplot in enumerate(self.instr.y_subplots):
            for self.context.j in range(len(y_subplot[self.context.idx])):
                if pd.isnull(self.instr.multigraph[self.context.i][self.context.idx]):#reshape dataframe for multiplot
                    pivot = pd.pivot_table(data_pivot, index = self.instr.x_axis[self.context.idx], values = y_subplot[self.context.idx][self.context.j])
                    self.plot_inner(pivot = pivot, label = y_subplot[self.context.idx][self.context.j],  axs = axs[self.context.i])
                    #label = tmp_label+" "+y_subplot[self.context.idx][self.context.j]
                else:
                    pivot = pd.pivot_table(data_pivot,
                                           columns = self.instr.multigraph[self.context.i][self.context.idx],
                                           index = self.instr.x_axis[self.context.idx],
                                           values = y_subplot[self.context.idx][self.context.j])
                    self.plot_inner(pivot = pivot, label = pivot.columns,  axs = axs[self.context.i])
    def plot_inner(self, pivot, label, axs):
        axs.plot(pivot.index, pivot.values, marker='.', label=label)
        axs.legend(loc=1,fontsize=7) #TODO magari 0
        if self.instr.vertical_lines is not None and self.instr.vertical_lines[self.context.idx] is not None:  
            lines = self.instr.vertical_lines[self.context.idx].split(",")
            for line in lines:
                axs.axvline(float(line))
        if self.instr.y_lower[self.context.i] is not None and pd.notnull(self.instr.y_lower[self.context.i][self.context.idx]):#Y-limits
            axs.set_ylim(bottom = self.instr.y_lower[self.context.i][self.context.idx])
        if self.instr.y_upper[self.context.i] is not None and pd.notnull(self.instr.y_upper[self.context.i][self.context.idx]):#Y-limits
            axs.set_ylim(top = self.instr.y_upper[self.context.i][self.context.idx])
        if self.instr.x_start is not None and pd.notnull(self.instr.x_start[self.context.idx]):
            axs.set_xlim(left = self.instr.x_start[self.context.idx])
        if self.instr.x_stop is not None and pd.notnull(self.instr.x_stop[self.context.idx]):
            axs.set_xlim(right = self.instr.x_stop[self.context.idx])
    def set_labels(self, page = None):
        self.mp.fig.supxlabel(self.instr.x_axis[self.context.idx]+" ("+self.instr.units[self.instr.x_axis[self.context.idx]]+")",fontsize=12,fontweight="bold")
        self.mp.fig.supylabel(self.instr.y_subplots[0][self.context.idx][0]+" ("+self.instr.units[self.instr.y_subplots[0][self.context.idx][0]]+")",fontsize=12,fontweight="bold")
        if page is not None:
            self.mp.fig.suptitle("%s"%self.instr.title[self.context.idx],fontsize=12,fontweight = 'bold')
        else:
            self.mp.fig.suptitle("%s"%self.instr.title[self.context.idx]+"  %s"%self.instr.page[self.context.idx]+"=%s"%page,fontsize=12,fontweight = 'bold')
    def remove_duplicate_order_list(self, el):
        return sorted(list(set(el)))
