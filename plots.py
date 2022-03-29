import math
from vamshiPlot import read_instruction as ri
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openpyxl
import pandas as pd
class MultiMultiplot:
    def __init__(self, width, height, n_rows, n_cols, n_subplots = 1):
        self.fig = plt.figure(figsize=(width, height))
        if len(n_subplots)>1:
            space = 0.1
        else:
            space = 0
        outer = gridspec.GridSpec(n_rows, n_cols, wspace=space, hspace=space)
        self.axs = []
        cont = 0
        for row in range(n_rows):
            col_axs = []
            for col in range(n_cols):
                sp_axs = []
                inner = gridspec.GridSpecFromSubplotSpec(n_subplots, 1,
                    subplot_spec=outer[cont], wspace=0, hspace=0)
                cont+=1
                for sp in range(n_subplots):
                    axs = plt.Subplot(self.fig, inner[sp])
                    self.fig.add_subplot(axs)
                    sp_axs.append(axs)
                col_axs.append(sp_axs)
            self.axs.append(col_axs)
class Plots:
    def __init__(self, instr):
        self.instr = instr
        self.pdf = PdfPages(self.instr.output_name+".pdf")
        self.width = instr.width
        self.height = instr.height
        self.plot()
    def plot(self):
        for idx in range(self.instr.length):
            pages = [" "]
            if self.instr.page is None or self.instr.page[idx].isnull():
                skip_pages = True
            else:
                skip_pages = False
                pages = self.remove_duplicate_order_list(self.instr.data[self.instr.page[idx]])
            for page in pages:
                self.data = self.instr.select_data(idx)
                if pd.isnull(self.instr.rows[idx]):
                    single_el = len(set(self.data[self.instr.columns[idx]]))
                    n_rows = int(math.sqrt(single_el))
                    n_cols = math.ceil(single_el/n_rows)
                else:
                    n_rows = len(set(self.data[self.instr.rows[idx]]))
                    n_cols = len(set(self.data[self.instr.columns[idx]]))
                self.mp = MultiMultiplot(width= self.instr.width, height = self.instr.height, n_rows = n_rows, n_cols = n_cols, n_subplots = self.instr.num_subplots)
                i_col = 0
                i_row = 0
                data = self.data
                if not skip_pages:
                    data = data[(data[self.instr.page[idx]]==page)]
                for col,c in enumerate(self.remove_duplicate_order_list(data[self.instr.columns[idx]])):
                    data = data[(data[self.instr.columns[idx]]==c)]
                    if pd.isnull(self.instr.rows[idx]):
                        self.plot_in_grid(data_pivot =data,
                                           idx = idx, i_col = i_col, i_row = i_row, tmp_label = "%s"%c)
                    i_col+=1
                    if i_col == n_cols:
                        i_col = 0
                        i_row+=1
                    else:
                        for row,r in enumerate(self.remove_duplicate_order_list(data[self.instr.columns[idx]])):
                            self.plot_in_grid(data_pivot = data[(data[self.instr.rows[idx]]==r)],
                                           idx = idx, i_col = col, i_row = row, tmp_label = "%s"%r+", %s"%c)
                if skip_pages:
                    self.set_labels(idx = idx)
                else:
                    self.set_labels(idx = idx, page = page)
                plt.show()
                self.pdf.savefig(self.mp.fig)
                plt.clf()
        self.pdf.close()
    def plot_in_grid(self, data_pivot, idx, i_col, i_row, tmp_label = None):
        for i, y_subplot in enumerate(self.instr.y_subplots):
            if pd.isnull(self.instr.multigraph[idx]):#reshape dataframe for multiplot
                pivot = pd.pivot_table(data_pivot,
                                       index = self.instr.x_axis[idx],
                                       values = y_subplot[idx])
                self.plot_inner(pivot = pivot, label = tmp_label, idx = idx,  axs = self.mp.axs[i_col][i_row][i])
            else:
                pivot = pd.pivot_table(data_pivot,
                                       columns = self.instr.multigraph[idx],
                                       index = self.instr.x_axis[idx],
                                       values = y_subplot[idx])
                self.plot_inner(pivot = pivot, label = pivot.columns, idx = idx,  axs = self.mp.axs[i_col][i_row][i])
    def plot_inner(self, pivot, label, idx, axs):
        axs.plot(pivot.index, pivot.values, marker='.', label=label)
        if self.instr.vertical_lines is not None and self.instr.vertical_lines[idx] is not None:  
            lines = self.instr.vertical_lines[idx].split(",")
            for line in lines:
                axs.axvline(float(line))
        axs.grid(axis='both',which='major')
        axs.tick_params(axis="both",which="major",direction="in",labelsize=4)
        axs.legend(loc=0,fontsize=7)
        if self.instr.y_lower is not None and pd.notnull(self.instr.y_lower[idx]):#Y-limits
            axs.set_ylim(bottom = self.instr.y_lower[idx])
        if self.instr.y_upper is not None and pd.notnull(self.instr.y_upper[idx]):#Y-limits
            axs.set_ylim(top = self.instr.y_upper[idx])
        if self.instr.range_i_x is not None and pd.notnull(self.instr.range_i_x[idx]):
            axs.set_xlim(left = self.instr.range_i_x[idx])
        if self.instr.range_f_x is not None and pd.notnull(self.instr.range_f_x[idx]):
            axs.set_xlim(right = self.instr.range_i_x[idx])
    def set_labels(self, idx, page = None):
        self.mp.fig.supxlabel(self.instr.x_axis[idx]+" ("+self.instr.units[self.instr.x_axis[idx]]+")",fontsize=12,fontweight="bold")
        self.mp.fig.supylabel(self.instr.y_axis[idx]+" ("+self.instr.units[self.instr.y_axis[idx]]+")",fontsize=12,fontweight="bold")
        if page is not None:
            self.mp.fig.suptitle("%s"%self.instr.title[idx],fontsize=12,fontweight = 'bold')
        else:
            self.mp.fig.suptitle("%s"%self.instr.title[idx]+"  %s"%self.instr.page[idx]+"=%s"%page,fontsize=12,fontweight = 'bold')
        #set xlabel
    def set_xlabel(self, axs, text):
        dims = axs.get_window_extent()
        pos = axs.get_position()
        self.mp.fig.text((dims.width/2+pos.width)*1.1,(dims.height/2+pos.height)*1.1, va='center')
    def remove_duplicate_order_list(self, el):
        return sorted(list(set(el)))
