import itertools
import matplotlib.pyplot as plt
import numpy as np
class Plot:
    def __init__(self, ext: str = "png"):
        self.ext =ext
        self.x = []
        self.y = []
        self.legend = []
        self.num_lines = 0
        self.vertical_lines = []
    def change_ext(self,ext):
        self.ext = ext
    def add_vertical_lines(self,line):
        self.vertical_lines.append(line)
    def add_line(self, x, y,legend = ''):
        self.x.append(x)
        self.y.append(y)
        self.num_lines += 1
        if legend is not None:
            self.legend.append(legend)
        else:
            self.legend.append('')
    def plot_data(self,
             x_label: str = '',
             x_um: str = '',
             plot_title: str = '',
             saveas: str = '',
             is_log: bool = False,
             show = True,
             save = True,
             show_legend = False,
             show_vertical_line = True,
             normalize = False
             ) -> str:
        for (x,y,legend) in zip(self.x,self.y,self.legend):
            
            if normalize:
                const = np.max(y)
            else:
                const = 1
            plt.plot(x,y/const, label = legend)
        if show_legend:
            plt.legend()
        if is_log:
            plt.yscale('log')
        plt.xlabel(x_label+"["+x_um+"]")
        plt.ylabel("counts [a.u.]")
        plt.title(plot_title)
        if show_vertical_line:
            for el in self.vertical_lines:
                plt.axvline(el)
        if save:
            plt.savefig(saveas+"." +self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    def multiple_line_subplots(self, y,y_um,
                                    xlabels, x_um,
                                    title, saveas,
                                    is_setLabel = True, show = False, 
                                    save = True, show_legend = False,
                                     show_vertical_line = True):
        sz = len(y)
        #TODO se è solo 1 cosa devo fare??
        fig, axs = plt.subplots(sz)
        fig.suptitle(title)
        for (x,data,legend) in zip(self.x,self,y,self.legend):
            for i in range(sz):
                axs[i].plot(x, data[i], label = legend)
                axs[i].tick_params(labelbottom=False)
                if is_setLabel:
                    axs[i].set_ylabel(str(int(y[i]*100)/100)+ " "+ y_um)
                if show_vertical_line:
                    for el in self.vertical_lines:
                        axs[i].axvline(el)
                if show_legend:
                    plt.legend()
        axs[sz-1].tick_params(labelbottom=True)
        plt.xlabel(xlabels + "["+ x_um+"]")
        plt.title(title)
        if save:
            plt.savefig(saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    #eccezioni
    def plot_data2D(self,
             x: np.array,
             y: np.array,
             data: np.array,
             x_label: str,
             x_um: str,
             y_label: str,
             y_um: str,
             plot_title: str,
             saveas: str,
             show = False,
             save = True ) -> str:
        plt.rcParams['pcolor.shading'] ='nearest'
        plt.pcolormesh(x, y,data)
        plt.xlabel( x_label+" ["+x_um+"]")
        plt.ylabel(y_label+ "["+y_um+"]")
        plt.title(plot_title)
        plt.colorbar()
        #TODO consigliabile non usare eps. perchè pesante in 2D!
        if save:
            plt.savefig(saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    def multiple_line_subplots_2(self, x, data, title, xlabels, x_um, saveas, show = False, save = True):
        sz = len(data)
        for i in range(sz):
            mx = max(data[i])
            off = i*1
            plt.plot(x, data[i]/mx+off)
        plt.yticks([])
        plt.xlabel(xlabels + "["+ x_um+"]")
        plt.title(title)
        if save:
            plt.savefig(saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
