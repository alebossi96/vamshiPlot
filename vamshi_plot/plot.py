import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import numpy as np
import math
import openpyxl
from sklearn.linear_model import LinearRegression

def multiplot_short(FileInstructions):
    
    scenario = pd.read_excel(FileInstructions+".xlsx",sheet_name="scenario")
    parameters = pd.read_excel(FileInstructions+".xlsx",sheet_name="parameters")
    variables = pd.read_excel(FileInstructions+".xlsx",sheet_name="variables")
    #User must change directory address
    if False:
        ref_spectra = pd.read_excel(r"C:\Users\Sanga\Desktop\OneDrive - Politecnico di Milano\Vamshi\Asus laptop\Reference spectra\Reference spectra.xlsx",sheet_name="data")
        linearity = pd.read_excel(r"C:\Users\Sanga\Desktop\OneDrive - Politecnico di Milano\Vamshi\Asus laptop\Reference spectra\CalibInk_03122019.xlsx",sheet_name="Data")
    
    #GET INPUT FROM PARAMETERS FILE
    field = parameters["Field"]
    value = parameters["Value"]
    PATH_IP = value[list(field).index("Input_data_dir")]
    FileData = value[list(field).index("Input_data")]
    PATH_OP = value[list(field).index("Save_into")]
    pdf_name = value[list(field).index("Pdf_name")]
    width = value[list(field).index("Width")]
    hieght = value[list(field).index("Hieght")]
    
    #GET INPUT FROM SCENARIO FILE
    section = scenario["Section"]
    page = scenario["Page"]
    title = scenario["Title"]
    select1 = scenario["Select1"]
    value1 = scenario["Value1"]
    select2 = scenario["Select2"]
    value2 = scenario["Value2"]
    Rows = scenario["Rows"]
    Columns = scenario["Columns"]
    row_title = scenario["Row_title"]
    col_title = scenario["Col_title"]
    reference = scenario["Reference"]
    multigraph = scenario["MultiGraph"]
    x_axis = scenario["X-axis"]
    y_axis = scenario["Y-axis"]
    y_scale = scenario["Scale"]
    y_lower = scenario["Y-lower"]
    y_upper = scenario["Y-upper"]
        
    #GET INPUT FROM VARIABLES FILE
    var_old_name = variables["Var_old_name"]
    var_new_name = variables["Var_new_name"]
    units = {}
    for i,u in zip(var_new_name,variables["Units"]):
        units[i]=u
       
    #GET DATA AS 'INDATA' AND CHANGE VARIABLE NAMES AND UNITS
    Indata = pd.read_csv(PATH_IP+FileData+".txt",delimiter=",")
    pdf = PdfPages(PATH_OP+pdf_name+".pdf")
    for old,new in zip(var_old_name,var_new_name):
        Indata = Indata.rename(columns={old:new})

    #CODE FOR PLOTTING
    for s in section:
        #RESHAPE DATAFRAME BASED ON 'PAGE' AND 'SELECT'
        for p in sorted(list(set(Indata[page[s]]))):
            if pd.isnull(select1[s]): 
                data1 = Indata
            else:
                data1 = Indata[Indata[select1[s]]==value1[s]]
            if pd.isnull(select2[s]): 
                data = data1
            else:
                data = data1[data1[select2[s]]==value2[s]]   
                
            #CODE FOR PLOTTING BY THE COLUMNS
            if pd.isnull(Rows[s]):
                row=0
                col=0
                nrows=int(math.sqrt(len(set(data[Columns[s]]))))#fix subplots array size
                ncols=math.ceil((len(set(data[Columns[s]])))/nrows)#fix subplots array size
                fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(width,hieght),squeeze=False)
                plt.subplots_adjust(hspace=0,wspace=0,left=0.09,bottom=0.09)
                fig.supxlabel(x_axis[s]+" ("+units[x_axis[s]]+")",fontsize=12,fontweight="bold")
                fig.supylabel(y_axis[s]+" ("+units[y_axis[s]]+")",fontsize=12,fontweight="bold")
                fig.suptitle("%s"%title[s]+"  %s"%page[s]+"=%s"%p,fontsize=12,fontweight = 'bold')            
                for i,c in enumerate(sorted(list(set(data[Columns[s]])))):
                    if pd.isnull(multigraph[s]):#reshape dataframe for multiplot
                        pivot = pd.pivot_table(data[(data[Columns[s]]==c)&
                                                    (data[page[s]]==p)],                                       
                                               index = x_axis[s],
                                               values = y_axis[s])
                        axs[row,col].plot(pivot.index,pivot.values,marker='.',label="%s"%c)
                    else:
                        pivot = pd.pivot_table(data[(data[Columns[s]]==c)&
                                                    (data[page[s]]==p)],
                                               columns = multigraph[s],
                                               index = x_axis[s],
                                               values = y_axis[s])
                        axs[row,col].plot(pivot.index,pivot.values,marker='.',label=pivot.columns)
                    if pd.isnull(reference[s]):#adding reference spectra
                        pass
                    elif reference[s]=="Linearity":
                        pivot_lin=pd.pivot_table(linearity[(linearity["Rho"]==1.5)&
                                                           (linearity["Accept"]=="OK")&
                                                           (linearity["sdev"]=="20x")&
                                                           (linearity["Exp"]=="EXP5")&
                                                           (linearity["Ink"]=="HIGG")&
                                                           (linearity["Lambda"]==c)],
                                                 index = "ConcInk",values="Mua")
                        model = LinearRegression().fit(np.array(pivot_lin.index).reshape((-1,1)),(pivot_lin.values))
                        axs[row,col].plot(np.array(pivot_lin.index).reshape((-1,1)),model.predict(np.array(pivot_lin.index).reshape((-1,1))))
                        #axs[row,col].plot(pivot_lin.index,pivot_lin.values,label="Theoretical")
                    elif reference[s]=='True Mua':
                        m=max(data[x_axis[s]])+0.1*max(data[x_axis[s]])
                        axs[row,col].plot(np.linspace(0,m,20),np.linspace(0,m,20))
                    elif reference[s]=="Ink":
                        ink_mua=[]
                        if Rows[s]=="Conc":
                            x=r
                        else:
                            x=c
                        for j,k in zip(ref_spectra.Lambda,ref_spectra.Water):
                            i=k+(x*3899.2*((j/600)**(-1.0997)))
                            ink_mua.append(i) 
                        axs[row,col].plot(ref_spectra.Lambda,ink_mua,label="Theoretical")
                    else:
                        for i in ref_spectra.columns:
                            if reference[s]==i:
                                axs[row,col].plot(ref_spectra.Lambda,ref_spectra[i],label=i)
                    axs[row,col].grid(axis='both',which='major')
                    axs[row,col].tick_params(axis="both",which="major",direction="in",labelsize=7)
                    axs[row,col].legend(loc=0,fontsize=7) 
                    if pd.isnull(y_lower[s]):#Y-limits
                        pass
                    else:
                        axs[row,col].set_ylim([y_lower[s],y_upper[s]])  
                    axs[row,col].set_xlim([min(data[x_axis[s]]),max(data[x_axis[s]])])
                    if (row_title[s] == True) and (col==0):
                        axs[row,col].set_ylabel("%s"%Columns[s]+" \n= " +"%s"%c)
                    if (col_title[s] == True) and (row==0):
                        axs[row,col].set_title("%s"%Columns[s]+" \n= "+"%s"%c)                
                    col=col+1
                    if col==ncols:
                        row=row+1
                        col=0
                plt.show()
                pdf.savefig(fig)
                plt.clf()
                                
            #CODE FOR PLOTTING BY ROWS AND COLUMNS: INDEPENDANTLY    
            else:
                fig,axs = plt.subplots(len(set(data[Rows[s]])),len(set(data[Columns[s]])),sharex=True,sharey=True,figsize=(width,hieght),squeeze=False)
                plt.subplots_adjust(hspace=0,wspace=0,left=0.09,bottom=0.09)
                fig.supxlabel(x_axis[s]+" ("+units[x_axis[s]]+")",fontsize=12,fontweight="bold")
                fig.supylabel(y_axis[s]+" ("+units[y_axis[s]]+")",fontsize=12,fontweight="bold")
                fig.suptitle("%s"%title[s]+"  %s"%page[s]+"=%s"%p,fontsize=12,fontweight = 'bold')
                for row,r in enumerate(sorted(list(set(data[Rows[s]])))):
                    for col,c in enumerate(sorted(list(set(data[Columns[s]])))):
                        if pd.isnull(multigraph[s]):
                            pivot = pd.pivot_table(data[(data[Columns[s]]==c)&
                                                        (data[Rows[s]]==r)&
                                                        (data[page[s]]==p)],
                                                   index=x_axis[s],
                                                   values=y_axis[s])
                            axs[row,col].plot(pivot.index,pivot.values,marker='.',label="%s"%r+", %s"%c)                            
                        else:
                            pivot = pd.pivot_table(data[(data[Columns[s]]==c)&
                                                        (data[Rows[s]]==r)&
                                                        (data[page[s]]==p)],
                                                   columns=multigraph[s],
                                                   index=x_axis[s],
                                                   values=y_axis[s])
                            axs[row,col].plot(pivot.index,pivot.values,marker='.',label=pivot.columns)
                        if pd.isnull(reference[s]):
                            pass
                        elif reference[s]=="Linearity":
                            pivot_lin=pd.pivot_table(linearity[(linearity["Rho"]==1.5)&
                                                               (linearity["Accept"]=="OK")&
                                                               (linearity["sdev"]=="20x")&
                                                               (linearity["Exp"]=="EXP5")&
                                                               (linearity["Ink"]=="HIGG")&
                                                               (linearity["Lambda"]==c)],
                                                     index = "ConcInk",values="Mua")
                            axs[row,col].plot(pivot_lin.index,pivot_lin.values,label="Theoretical")
                        elif reference[s]=="Ink":
                            ink_mua=[]
                            if Rows[s]=="Conc":
                                x=r
                            else:
                                x=c
                            for j,k in zip(ref_spectra.Lambda,ref_spectra.Water):
                                i=k+(x*3899.2*((j/600)**(-1.0997)))
                                ink_mua.append(i) 
                            axs[row,col].plot(ref_spectra.Lambda,ink_mua,label="Theoretical")
                        elif reference[s]=='True Mua':
                            m=max(data[x_axis[s]])+0.1*max(data[x_axis[s]])
                            axs[row,col].plot(np.linspace(0,m,20),np.linspace(0,m,20),'--',label="True Mua")
                        else:
                            for i in ref_spectra.columns:
                                if reference[s]==i:
                                    axs[row,col].plot(ref_spectra.Lambda,ref_spectra[i],label=i)
                        axs[row,col].tick_params(axis="both",which="major",direction="in",labelsize=7)
                        axs[row,col].legend(loc=0,fontsize=7)
                        axs[row,col].grid(axis='both',which='major')
                        if pd.isnull(y_lower[s]):
                            pass
                        else:       
                            axs[row,col].set_ylim([y_lower[s],y_upper[s]])   
                        axs[row,col].set_xlim([min(data[x_axis[s]]),max(data[x_axis[s]])])
                        axs[row,col].legend(loc=0,fontsize=7)
                        axs[row,col].tick_params(axis="both",which="major",direction="in",labelsize=7)
                        if (row_title[s] == True) and (col==0):
                            axs[row,col].set_ylabel("%s"%Rows[s]+" \n= " +"%s"%r)
                        if (col_title[s] == True) and (row==0):
                            axs[row,col].set_title("%s"%Columns[s]+" \n= " +"%s"%c)   
                        axs[row,col].set_yscale(y_scale[s])
                plt.show()
                pdf.savefig(fig)
                plt.clf()
    pdf.close()
    
