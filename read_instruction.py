import math
import pandas as pd
import numpy as np
class ReadInstruction:
    def __init__(self, file_instruction):
        #read what to plot
        scenario = pd.read_excel(file_instruction+".xlsx",sheet_name="scenario")
        #read the input? and output file
        parameters = pd.read_excel(file_instruction+".xlsx",sheet_name="parameters")
        #change names
        variables = pd.read_excel(file_instruction+".xlsx",sheet_name="variables")
        #GET INPUT FROM PARAMETERS FILE
        field = parameters["Field"]
        value = parameters["Value"]
        path_op = value[list(field).index("Save_into")]
        pdf_name = value[list(field).index("Pdf_name")]
        self.output_name = path_op + pdf_name
        #forse le dovrei mettere come costanti fuori da qui
        self.width = value[list(field).index("Width")]
        self.hieght = value[list(field).index("Hieght")]
        #GET INPUT FROM SCENARIO FILE
        self.index = scenario["Section"]
        self.page = scenario["Page"]
        self.title = scenario["Title"]
        self.num_selection = 0
        for el in scenario.columns:
            if el == "Select" +str(self.num_selection+1):
                self.num_selection+=1
        #Select and the others are lists(they could be an objects) because we want to have n possible selection
        self.select = []
        self.value = []
        self.range_select_min = []
        self.range_select_max = []
        self.subsample = []
        for i in range(self.num_selection):
            for el in scenario.columns:
                if "Select"+str(i+1) == el:
                    self.select.append(scenario["Select"+str(i+1)])
                elif "Value"+str(i+1) == el:
                    self.value.append( scenario["Value"+str(i+1)])
                elif "Range_min"+str(i+1) == el:
                    self.range_select_min.append(scenario["Range_min"+str(i+1)])
                elif "Range_max"+str(i+1) == el:
                    self.range_select_max.append(scenario["Range_max"+str(i+1)])
                elif "Subsample"+str(i+1) == el:
                    self.subsample.append(scenario["Subsample"+str(i+1)])
                
            if len(self.select)> len(self.value):
                self.value.append(['']*len(self.index))
            if len(self.select)> len(self.range_select_min):
                self.range_select_min.append(['']*len(self.index))
            if len(self.select)> len(self.range_select_max):
                self.range_select_max.append(['']*len(self.index))
            if len(self.select)> len(self.subsample):
                self.subsample.append([1]*len(self.index))
        self.rows = scenario["Rows"]
        self.columns = scenario["Columns"]
        self.row_title = scenario["Row_title"]
        self.col_title = scenario["Col_title"]
        self.reference = scenario["Reference"]
        self.multigraph = scenario["MultiGraph"]
        self.x_axis = scenario["X-axis"]
        try:
            self.x_start = scenario["Range_init_x"]
        except KeyError:
            self.x_start = None
        try:
            self.x_stop = scenario["Range_fin_x"]
        except KeyError:
            self.x_stop = None
            
        #self.y_axis = scenario["Y-axis"]
        self.num_subplots = 1
        self.y_subplots = []
        self.y_subplots.append(scenario["Y-axis"])
        for el in scenario.columns:
            #I could want to have multiple graph to show on a single step.
            if el == "Y-axis" +str(self.num_subplots+1):
                self.num_subplots+=1
        for i in range(self.num_subplots):
            for el in scenario.columns:
                if "Y-axis"+str(i+1) == el:
                    self.y_subplots.append(scenario["Y-axis"+str(i+1)])
        self.y_scale = scenario["Scale"]
        self.y_lower = scenario["Y-lower"]
        self.y_upper = scenario["Y-upper"]
        #Se voglio mettere cose a dx devo comunque lasciare lon stesso asse x!
        try:
            self.y_right = scenario["Y-axis-right"]
        except:
            self.y_right = None
        try:
            self.y_scale_right = scenario["Scale-right"]
        except:
            self.y_scale_right = None
        try:
            self.y_lower_right = scenario["Y_lower-right"]
        except:
            self.y_lower_right = None
        try:
            self.y_upper_right = scenario["Y_upper-right"]
        except:
            self.y_upper_right = None
        try:
            self.vertical_lines = scenario["Vertical_lines"]
        except KeyError:
            self.vertical_lines = None
        #GET INPUT FROM VARIABLES FILE
        var_old_name = variables["Var_old_name"]
        var_new_name = variables["Var_new_name"]
        self.units = {}
        for i,unit in zip(var_new_name,variables["Units"]):
            self.units[i]=unit
        self.data = self.get_data()
        for old,new in zip(var_old_name,var_new_name):
            self.data = self.data.rename(columns={old:new})
    def refactor_data(self, idx):
        data = self.data
        for i in range(self.num_selection):
            data = self.select_data(data, idx, i)
        return data
    def select_data(self,data, idx, i):
        """
        Selects the data in a given range.
        per ora funziona solo uno alla volta
        """
        if pd.isnull(self.select[idx][i]):
            return data
        if self.subsample[idx][i] != 1 :
            pivot = pd.pivot_table(data,index=[self.select[idx]])
            index = pivot[self.select[idx]]
            subsampling = np.arange(0,len(pivot), self.subsample[idx][i])
            data = data[index == index[subsampling]]
            return data
            
        if self.value[idx][i] != '':
            return data[data[self.select[idx][i]]==self.value[idx][i]]
        if self.range_select_min[idx][i] != '' and self.range_select_max[idx][i] != '':
            return data[(data[self.select[idx][i]]>self.range_select_min[idx][i]) & (data[self.select[idx][i]]<self.range_select_max[idx][i])]
        if self.range_select_min[idx][i] != '':
            return data[data[self.select[idx][i]]>self.range_select_min[idx][i]]
        if self.range_select_max[idx][i] != '':
            return data[data[self.select[idx][i]]<self.range_select_max[idx][i]]
                
        raise TypeError("you must enter a value or range to be selected")

class GetDataFromTXT(ReadInstruction):
    def __init__(self, file_instruction):
        parameters = pd.read_excel(file_instruction+".xlsx",sheet_name="parameters")
        field = parameters["Field"]
        value = parameters["Value"]
        PATH_IP = value[list(field).index("Input_data_dir")]
        FileData = value[list(field).index("Input_data")]
        self.data_pos = PATH_IP+FileData
        super().__init__(file_instruction)
    def get_data(self):
        return pd.read_csv(self.data_pos+".txt", delimiter =',')#magari devo leggere
class GetDataNumpy(ReadInstruction):
    def __init__(self, file_instruction, data_numpy, data_titles):
        self.data_numpy = data_numpy
        self.data_titles = data_titles
        super().__init__(file_instruction)
    def get_data(self):
        return  pd.DataFrame(self.data_numpy, columns = self.data_titles)
