from librerieTesi.reconstructTDDRV2 import RAST, HAD, f_tT_, reconstructTDDR
import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
class AllignementTDSpectrometer:
    def __init__(self, filename, tot_num_banks, nBasis, rastOrHad, time_per_base):
        self.filename = filename
        self.tot_num_banks = tot_num_banks
        self.nBasis = nBasis
        self.rastOrHad = rastOrHad
        self.time_per_base = time_per_base
    def readSdt(self, fileName: str, pos_bank : int) -> np.array:
        data = []
        zeros = int(np.log10(self.tot_num_banks+1))
        fn = fileName
        if self.tot_num_banks == 1:
            sdt = sdtfile.SdtFile(fn+".sdt")
        elif pos_bank==0:       
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(pos_bank)+".sdt")
        elif pos_bank<10:       
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(pos_bank)+".sdt")
        elif pos_bank<100:       
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-1)+str(pos_bank)+".sdt")
        elif pos_bank<1000:       
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-2)+str(pos_bank)+".sdt")
        elif pos_bank<10000:       
            sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-3)+str(pos_bank)+".sdt")
        batch = sdt.data[0].shape[0];
        for j in range(batch):
            data.append(sum(sdt.data[0][ j,:]))
        data = np.array(data)
        return data
    def execute(self):
        for i in range(self.tot_num_banks):
            data = self.readSdt(self.filename, i)
            rec =  reconstructTDDR(data = data,
                                    rastOrHad = self.rastOrHad,
                                    lambda_0 = 780,
                                    )                 
    def plot(self):
        pass
        #TODO avere un modo per aggiornare l'immagine in plt.show()
