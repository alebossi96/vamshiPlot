class Selection:
    def __init__(self, i, column, scenario):
        self.name = None
        self.value = None
        self.range_select_min = None
        self.range_select_max = None
        self.subsample = None
        for el in column:
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
            
