from typing import List, Tuple
import numpy as np
import  sdtfile

def readSdt(self, filename: str, tot_banks = None ) -> Tuple[np.array,List]:
    data = []
    for i in range(self.n_banks):
        if tot_banks:
            zeros = int(np.log10(tot_banks+1))
        else:
            zeros = int(np.log10(self.n_banks+1))
        fn = filename
        if self.n_banks == 1:
            sdt = sdtfile.SdtFile(fn+".sdt")
        elif i==0:
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
        elif i<10:
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
        elif i<100:
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-1)+str(i)+".sdt")
        elif i<1000:
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-2)+str(i)+".sdt")
        elif i<10000:
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-3)+str(i)+".sdt")
        time = sdt.times[0][:]
        batch = sdt.data[0].shape[0]
        for j in range(batch):
            data.append(sdt.data[0][ j,:])
    return (time,data)
    
def time_to_idx(array,time):
    i = 0
    while array[i]<time:
        i+=1
    return i
