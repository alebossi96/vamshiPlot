import math
import pandas as pd
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import openpyxl
class Multiplot:
    def __init__(self, FileInstructions):
        scenario = pd.read_excel(FileInstructions+".xlsx",sheet_name="scenario")
        parameters = pd.read_excel(FileInstructions+".xlsx",sheet_name="parameters")
        variables = pd.read_excel(FileInstructions+".xlsx",sheet_name="variables")
        #GET INPUT FROM PARAMETERS FILE
        field = parameters["Field"]
        value = parameters["Value"]
        PATH_IP = value[list(field).index("Input_data_dir")]
        FileData = value[list(field).index("Input_data")]
        PATH_OP = value[list(field).index("Save_into")]
        pdf_name = value[list(field).index("Pdf_name")]
        self.width = value[list(field).index("Width")]
        self.hieght = value[list(field).index("Hieght")]
        #GET INPUT FROM SCENARIO FILE
        self.index = scenario["Section"]
        self.page = scenario["Page"].fillna(" ")
        self.title = scenario["Title"]
        try:
            self.range_i_x = scenario["Range_init_x"]#catch no range field
        except KeyError:
            self.range_i_x = None
        try:
            self.range_f_x = scenario["Range_fin_x"]
        except KeyError:
            self.range_f_x = None
        self.select1 = scenario["Select1"]#custom possibilities for number of selection
        self.value1 = scenario["Value1"]
        self.select2 = scenario["Select2"]
        self.value2 = scenario["Value2"]
        self.Rows = scenario["Rows"]
        self.Columns = scenario["Columns"]
        self.row_title = scenario["Row_title"]
        self.col_title = scenario["Col_title"]
        self.reference = scenario["Reference"]
        self.multigraph = scenario["MultiGraph"]
        self.x_axis = scenario["X-axis"]
        self.y_axis = scenario["Y-axis"]
        self.y_scale = scenario["Scale"]
        self.y_lower = scenario["Y-lower"]
        self.y_upper = scenario["Y-upper"]
        #GET INPUT FROM VARIABLES FILE
        var_old_name = variables["Var_old_name"]
        var_new_name = variables["Var_new_name"]
        self.units = {}
        for i,u in zip(var_new_name,variables["Units"]):
            self.units[i]=u
        #read data
        self.Indata = self.read_data(filename = PATH_IP+FileData+".txt")
        for old,new in zip(var_old_name,var_new_name):
            self.Indata = self.Indata.rename(columns={old:new})
        self.pdf = PdfPages(PATH_OP+pdf_name+".pdf")
        self.plot()   
    def read_data(self, filename = None, data_numpy = None, data_titles = None):
        if filename is not None:
            #TODO mettere caso excel
            return pd.read_csv(filename,delimiter="\t")
        if data_numpy is not None and data_titles is not None:
            return pd.DataFrame(data_numpy, columns = data_titles)
        raise TypeError("you must enter some data to be parsed")
    def select_data(self, data_in, select, value):
        if pd.isnull(select):
            return data_in
        return data_in[data_in[select]==value]
    def plot(self):
        for idx in self.index:
            if self.page[idx] == " ":
                pages = [" "]
                skip_pages = True
            else:
                pages = self.remove_duplicate_order_list(self.Indata[self.page[idx]])
                skip_pages = False
            for page in pages:
                data1 = self.select_data(data_in = self.Indata, select = self.select1[idx], value = self.value1[idx])
                self.data = self.select_data(data_in = data1, select = self.select2[idx], value = self.value2[idx])
                if pd.isnull(self.Rows[idx]):
                    row=0
                    col=0
                    nrows=int(math.sqrt(len(set(self.data[self.Columns[idx]]))))#fix subplots array size
                    ncols=math.ceil((len(set(self.data[self.Columns[idx]])))/nrows)#fix subplots array size
                    fig,axs = self.gen_page(nrows, ncols, idx, page)
                    for c in self.remove_duplicate_order_list(self.data[self.Columns[idx]]):
                        if skip_pages:
                            self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)],
                                          axs = axs[row, col], c = c, idx = idx, col = col, row = row)
                        else:
                            self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                         (self.data[self.page[idx]]==page)],
                                          axs = axs[row, col], c = c, idx = idx, col = col, row = row)
                        col=col+1
                        if col==ncols:
                            row=row+1
                            col=0
                #CODE FOR PLOTTING BY ROWS AND COLUMNS: INDEPENDENTLY    
                else:
                    fig,axs = self.gen_page(nrows = len(set(self.data[self.Rows[idx]])),
                                                     ncols = len(set(self.data[self.Columns[idx]])),
                                                                 idx = idx,
                                                                 page = page)
                    for row,r in enumerate(self.remove_duplicate_order_list(self.data[self.Rows[idx]])):
                        for col,c in enumerate(self.remove_duplicate_order_list(self.data[self.Columns[idx]])):
                            if skip_pages:
                                self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                             (self.data[self.Rows[idx]]==r)],
                                          axs = axs[row, col], c = c, r = r,idx = idx, col= col, row = row )
                            else:
                                self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                             (self.data[self.Rows[idx]]==r)&
                                             (self.data[self.page[idx]]==page)],
                                          axs = axs[row, col], c = c, r = r,idx = idx, col= col, row = row )
                plt.show()
                self.pdf.savefig(fig)
                plt.clf()
        self.pdf.close()
    def gen_page(self,nrows, ncols, idx, page):
        fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(self.width,self.hieght),squeeze=False)
        plt.subplots_adjust(hspace=0.01,wspace=0.01,left=0.09,bottom=0.09)
        fig.supxlabel(self.x_axis[idx]+" ("+self.units[self.x_axis[idx]]+")",fontsize=12,fontweight="bold")
        fig.supylabel(self.y_axis[idx]+" ("+self.units[self.y_axis[idx]]+")",fontsize=12,fontweight="bold")
        fig.suptitle("%s"%self.title[idx]+"  %s"%self.page[idx]+"=%s"%page,fontsize=12,fontweight = 'bold')
        return (fig, axs)
    """
    def get_index_range(self,index, idx):
        range_low = 1
        range_high = 1
        if self.range_i_x is not None:
            range_low = (index>self.range_i_x[idx])
        if self.range_f_x is not None:
            range_high = (index<self.range_f_x[idx])
        return range_low & range_high
    """
    def plot_single(self, data_pivot, axs, c, idx, col, row, r = None):
        if pd.isnull(self.multigraph[idx]):#reshape dataframe for multiplot
            pivot = pd.pivot_table(data_pivot,
                                   index = self.x_axis[idx],
                                   values = self.y_axis[idx])
            if r is None:
                axs.plot(pivot.index,pivot.values,marker='.',label="%s"%c)
            else:
                axs.plot(pivot.index,pivot.values,marker='.',label="%s"%r+", %s"%c)
        else:
            pivot = pd.pivot_table(data_pivot,
                                   columns = self.multigraph[idx],
                                   index = self.x_axis[idx],
                                   values = self.y_axis[idx])
            axs.plot(pivot.index,pivot.values,marker='.',label=pivot.columns)
        axs.grid(axis='both',which='major')
        axs.tick_params(axis="both",which="major",direction="in",labelsize=7)
        axs.legend(loc=0,fontsize=7)
        if pd.isnull(self.y_lower[idx]):#Y-limits
            pass
        else:
            axs.set_ylim([self.y_lower[idx],self.y_upper[idx]])
            axs.set_xlim(left = self.range_i_x[idx], right = self.range_f_x[idx])
        if self.row_title[idx] is True and col==0:
            axs.set_ylabel("%s"%self.Columns[idx]+" \n= " +"%s"%c)
        if self.col_title[idx] is True and row==0:
            axs.set_title("%s"%self.Columns[idx]+" \n= "+"%s"%c)
        self.add_subplot_border(axs, width = 2, color = 'b')
            
    def add_subplot_border(self,ax, width=1, color=None ):

        fig = ax.get_figure()

        # Convert bottom-left and top-right to display coordinates
        x0, y0 = ax.transAxes.transform((0, 0))
        x1, y1 = ax.transAxes.transform((1, 1))

        # Convert back to Axes coordinates
        x0, y0 = ax.transAxes.inverted().transform((x0, y0))
        x1, y1 = ax.transAxes.inverted().transform((x1, y1))

        rect = plt.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            color=color,
            transform=ax.transAxes,
            zorder=-1,
            lw=2*width+1,
            fill=None,
        )
        fig.patches.append(rect)  
    def remove_duplicate_order_list(self, el):
        return sorted(list(set(el)))

