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
        self.PATH_IP = value[list(field).index("Input_data_dir")]
        self.FileData = value[list(field).index("Input_data")]
        PATH_OP = value[list(field).index("Save_into")]
        pdf_name = value[list(field).index("Pdf_name")]
        self.width = value[list(field).index("Width")]
        self.hieght = value[list(field).index("Hieght")]
        #GET INPUT FROM SCENARIO FILE
        self.index = scenario["Section"]
        self.page = scenario["Page"]
        self.title = scenario["Title"]
        self.select1 = scenario["Select1"]
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
        self.Indata = self.read_data()
        for old,new in zip(var_old_name,var_new_name):
            self.Indata = self.Indata.rename(columns={old:new})
        self.pdf = PdfPages(PATH_OP+pdf_name+".pdf")
        self.plot()   
    def read_data(self):
        return pd.read_csv(self.PATH_IP+self.FileData+".txt",delimiter=",")
    def select_data(data_in, select, value):
        if pd.isnull(self.select1[idx]):
            return data_in
        return data[data[select]==value]
    def plot(self):
        for idx in self.index:
            pages = self.remove_duplicate_order_list(self.Indata[self.page[idx]])
            for page in pages:
                data1 = self.select_data(data_in = self.Indata, select = self.select1[idx], value = self.value1[idx])
                data = self.select_data(data_in = data1, select = self.select2[idx], value = self.value2[idx])
                if pd.isnull(self.Rows[idx]):
                    row=0
                    col=0
                    nrows=int(math.sqrt(len(set(self.data[self.Columns[idx]]))))#fix subplots array size
                    ncols=math.ceil((len(set(self.data[self.Columns[idx]])))/nrows)#fix subplots array size
                    fig,axs = self.gen_fig(nrows, ncols, idx, page)
                    for c in self.remove_duplicate_order_list(self.data[self.Columns[idx]]):
                        self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                         (self.data[self.page[idx]]==page)] , axs = axs[row, col], c = c, idx = idx, col = col, row = row)
                        col=col+1
                        if col==ncols:
                            row=row+1
                            col=0
                #CODE FOR PLOTTING BY ROWS AND COLUMNS: INDEPENDENTLY    
                else:
                    fig,axs = fig,axs = self.gen_fig(nrows = len(set(self.data[self.Rows[idx]])),
                                                     ncols = len(set(self.data[self.Columns[idx]])),
                                                                 idx = idx,
                                                                 page = page)
                    for row,r in enumerate(self.remove_duplicate_order_list(self.data[self.Rows[idx]])):
                        for col,c in enumerate(self.remove_duplicate_order_list(self.data[self.Columns[idx]])):
                            self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                             (self.data[self.Rows[idx]]==r)&
                                             (self.data[self.page[idx]]==page)], axs = axs[row, col], c = c, r = r,idx = idx, col= col, row = row )
                plt.show()
                self.pdf.savefig(fig)
                plt.clf()
        self.pdf.close()
    def gen_fig(self,nrows, ncols, idx, page):
        fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(self.width,self.hieght),squeeze=False)
        plt.subplots_adjust(hspace=0,wspace=0,left=0.09,bottom=0.09)
        fig.supxlabel(self.x_axis[idx]+" ("+self.units[self.x_axis[idx]]+")",fontsize=12,fontweight="bold")
        fig.supylabel(self.y_axis[idx]+" ("+self.units[self.y_axis[idx]]+")",fontsize=12,fontweight="bold")
        fig.suptitle("%s"%self.title[idx]+"  %s"%self.page[idx]+"=%s"%page,fontsize=12,fontweight = 'bold')
        return (fig, axs)
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
            axs.set_xlim([min(self.data[self.x_axis[idx]]),max(self.data[self.x_axis[idx]])])
        if self.row_title[idx] is True and col==0:
            axs.set_ylabel("%s"%self.Columns[idx]+" \n= " +"%s"%c)
        if self.col_title[idx] is True and row==0:
            axs.set_title("%s"%self.Columns[idx]+" \n= "+"%s"%c)
    def remove_duplicate_order_list(self, el):
        return sorted(list(set(el)))

