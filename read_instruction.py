import math
import pandas as pd
import numpy as np
from vamshiPlot import core
from vamshiPlot import refactor
class ReadInstruction:
    def __init__(self, file_instruction):
        #read what to plot
        scenario = pd.read_excel(file_instruction+".xlsx",sheet_name="scenario")
        #read the input? and output file
        parameters = pd.read_excel(file_instruction+".xlsx",sheet_name="parameters")
        #change names
        variables = pd.read_excel(file_instruction+".xlsx",sheet_name="variables")
        #GET INPUT FROM PARAMETERS FILE
        #field = parameters["Field"]
        #value = parameters["Value"]
        path_op = parameters["Save_into"][0]#value[list(field).index("Save_into")]
        pdf_name = parameters["Pdf_name"][0]#value[list(field).index("Pdf_name")]
        self.output_name = path_op + pdf_name
        #forse le dovrei mettere come costanti fuori da qui
        self.width = parameters["Width"][0]# value[list(field).index("Width")]
        self.height = parameters["Height"][0]#value[list(field).index("Height")]
        #GET INPUT FROM SCENARIO FILE
        self.length = len(scenario)
        self.page = scenario["Page"]
        self.title = scenario["Title"]
        self.num_selection = 0
        for el in scenario.columns:
            if el == "Select" +str(self.num_selection+1):
                self.num_selection+=1
        #Select and the others are lists(they could be an objects) because we want to have n possible selection
        sfi_tmp = []
        for i in range(self.num_selection):
            sfi_tmp.append(core.SelectionFromInstruction(i, scenario.columns, scenario))
        self.sfi = core.invert_selection_instr(sfi_tmp)
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
            if el == "Y-axis" +str(self.num_subplots):
                self.num_subplots+=1
        for i in range(self.num_subplots):
            for el in scenario.columns:
                if "Y-axis"+str(i+1) == el:
                    self.y_subplots.append(scenario["Y-axis"+str(i+1)])
        self.y_scale = scenario["Scale"]
        self.y_lower = []#scenario["Y-lower"]
        self.y_upper = []#scenario["Y-upper"]
        self.y_lower.append(scenario["Y-lower"])
        self.y_upper.append(scenario["Y-upper"])
        for i in range(self.num_subplots):
            for el in scenario.columns:
                if "Y-lower"+str(i+1) == el:
                    self.y_lower.append(scenario["Y-lower"+str(i+1)])
                if "Y-upper"+str(i+1) == el:
                    self.y_upper.append(scenario["Y-upper"+str(i+1)])
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
            data = self.select_data(idx)
        return data
    def select_data(self, idx):
        """
        Selects the data in a given range.
        per ora funziona solo uno alla volta
        """
        data = self.data
        for sel in self.sfi[idx]:
            data = sel.select_data(data)
        return data
class GetDataFromTXT(ReadInstruction):
    def __init__(self, file_instruction, delimiter = ","):
        parameters = pd.read_excel(file_instruction+".xlsx",sheet_name="parameters")
        #field = parameters["Field"]
        #value = parameters["Value"]
        self.PATH_IP = parameters["Input_data_dir"][0]#value[list(field).index("Input_data_dir")]
        self.file_data = parameters["Input_data"] #value[list(field).index("Input_data")]
        if len(self.file_data)>1:
            self.axis_merge = parameters["Axis_merge"]
        try:
            self.refactor = parameters["Refactor"]
        except:
            self.refactor = None
        #self.data_pos = PATH_IP+FileData
        self.delimiter = delimiter
        super().__init__(file_instruction)
    def get_data(self):
        data_pos = self.PATH_IP+ self.file_data[0]
        df = pd.read_csv(data_pos, delimiter = core.get_delimiter(data_pos))
        if len(self.file_data) == 1:
            return df
        for i in range(1,len(self.file_data)):
            data_pos = self.PATH_IP+self.file_data[i]
            new = pd.read_csv(data_pos, delimiter = core.get_delimiter(data_pos))
            if self.refactor is not None and self.refactor.notnull()[i]:
                new = getattr(refactor, self.refactor[i])(new, self.axis_merge[i])
            df = df.merge(new, on = self.axis_merge[i], how= "inner")#magari devo leggere
        return df

class GetDataNumpy(ReadInstruction):
    def __init__(self, file_instruction, data_numpy, data_titles):
        self.data_numpy = data_numpy
        self.data_titles = data_titles
        super().__init__(file_instruction)
    def get_data(self):
        return  pd.DataFrame(self.data_numpy, columns = self.data_titles)
