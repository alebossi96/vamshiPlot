import math
import pandas as pd
import numpy as np
class SelectionFromInstruction:
    def __init__(self, i, column, scenario):
        self.select = None
        self.value = None
        self.range_select_min = None
        self.range_select_max = None
        self.subsample = None
        for el in column:
            #mettere qui che non voglio ''
            if "Select"+str(i+1) == el:
                self.select = scenario["Select"+str(i+1)]
            elif "Value"+str(i+1) == el:
                self.value = scenario["Value"+str(i+1)]
            elif "Range_min"+str(i+1) == el:
                self.range_select_min = scenario["Range_min"+str(i+1)]
            elif "Range_max"+str(i+1) == el:
                self.range_select_max = scenario["Range_max"+str(i+1)]
            elif "Subsample"+str(i+1) == el:
                self.subsample = scenario["Subsample"+str(i+1)]
class Selection:
    def __init__(self, sfi, idx):
        self.select = None
        self.value = None
        self.range_select_min = None
        self.range_select_max = None
        self.subsample = None
        if sfi.select is not None and sfi.select[idx] != '':
            self.select = sfi.select[idx]
        if sfi.value is not None and sfi.value[idx] != '':
            self.value = sfi.value[idx]
        if sfi.range_select_min is not None and sfi.range_select_min[idx] != '':
            self.range_select_min = sfi.range_select_min[idx]
        if sfi.range_select_max is not None and sfi.range_select_max[idx] != '':
            self.range_select_max = sfi.range_select_max[idx]
        if sfi.subsample is not None and sfi.subsample[idx]!= '':
            self.subsample = sfi.subsample[idx]
    def select_data(self, data):
        """
        Deve essere solo una delle 4!
        """
        """
        if self.subsample is not None :
            
            pivot = pd.pivot_table(data,index=[self.select])
            index = pivot.index
            subsampling = np.arange(0,len(pivot), self.subsample)
            to_take = [False]*len(data[self.select])
            for sub_smpl in range(len(subsampling)):
                tmp = [data[self.select][i] == index[subsampling[sub_smpl]] for i in range(len(to_take)) ]
                to_take = [to_take[i] == tmp[i] for i in range(len(to_take))] 
            data = data[to_take]
            
            
            delta = 1
            pivot = pd.pivot_table(data,index=[self.select])
            data = data.groupby(self.select).nth(slice(0,None, self.subsample*delta))
        """
        if self.value is not None:
            data = data[data[self.select]==self.value]
        if self.range_select_min is not None and self.range_select_max is not None:
            data = data[(data[self.select]>self.range_select_min) & (data[self.select]<self.range_select_max)]
        if self.range_select_min is not None:
            data = data[data[self.select]>self.range_select_min]
        if self.range_select_max is not None:
            data =  data[data[self.select]<self.range_select_max]
        return data
def invert_selection_instr(sfi):
    i_len = len(sfi)
    idx_len = len(sfi[0].select)
    sel = []
    for idx in range(idx_len):
        sel_el = []
        for i in range(i_len):
            if sfi[i].select[idx] is not None:
                sel_el.append(Selection(sfi[i], idx))
        sel.append(sel_el)
    return sel
           
