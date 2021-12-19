import numpy as np
import csv
#import List
class ExportSpectra:
    def __init__(self, x, counts):
        self.x = x
        self.y = counts
        self.length = len(self.x)
        self.num_t_gates = len(self.y)
    def title(self, x_exp, y_title):
        self.x_exp  = x_exp
        self.y_title = y_title
    def add_first_line(self):
        line = []
        line.append(self.x_exp)
        for el in self.y_title:
            print(el)
            line.append(el)
        return line
    def exportCsv(self, filename, first_line = False):
        with open(filename + ".csv", mode = 'w') as csv_f:
            writer = csv.writer(csv_f)
            if first_line:
                writer.writerow(self.add_first_line())
            for i in range(self.length):
                line = [self.x[i]]
                for j in range(self.num_t_gates):
                    line.append(self.y[j][i])
                writer.writerow(line) 
    
