# Take a look at the images stored by the short- and long-term memories

# by Stephane Vujasinovic
# TODO: ALSO SAVE THE INDEX VALUES INA JSON FILE

import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
from icecream import ic
import time

def gram_det_plot(name, data_y):
    plt.plot(data_y, color='r')
    plt.savefig(name)
    plt.clf()

class ST_LT_plotting_functions():
    def __init__(self, size_of_ST_Mem:int, size_of_LT_Mem:int):
        self.size_ST_Mem = size_of_ST_Mem
        self.size_LT_Mem = size_of_LT_Mem

    @staticmethod
    def add_blank_images(size_Mem:int, Mem:list, anno_img:np.array, indexes:list):
        if size_Mem != len(Mem):
            diff_size = size_Mem - len(Mem)

            blank_img_in_memory = [np.zeros(anno_img.shape) for _ in range(0,diff_size)]
            blank_indexes = ['None' for _ in range(0,diff_size)]

            Mem += blank_img_in_memory
            indexes += blank_indexes

        return Mem, indexes

    def set_path_for_finding_the_images(self, path:str):
        self.path = path

    def load_relevant_images(self, ST_memory_idx:list, LT_memory_idx:list):
        ic(ST_memory_idx,LT_memory_idx)
        # load image paths
        list_of_img_paths = [os.path.join(self.path,f) for f in os.listdir(os.path.join(os.getcwd(),self.path))
                             if os.path.isfile(os.path.join(os.getcwd(),self.path,f))]

        # load image paths
        self.ST_memory_img = [np.array(PIL.Image.open(list_of_img_paths[ST_idx])) for ST_idx in ST_memory_idx]
        self.LT_memory_img = [np.array(PIL.Image.open(list_of_img_paths[LT_idx])) for LT_idx in LT_memory_idx]

        # Check current size with max size defined
        self.ST_memory_img,self.ST_memory_idx = self.add_blank_images(self.size_ST_Mem,
                                                                      self.ST_memory_img,
                                                                      self.LT_memory_img[0],
                                                                      ST_memory_idx)
        self.LT_memory_img,self.LT_memory_idx = self.add_blank_images(self.size_LT_Mem,
                                                                      self.LT_memory_img,
                                                                      self.LT_memory_img[0],
                                                                      LT_memory_idx)

    def plt_ST_and_LT_memory(self):#, img_w_mask:np.array):
        # self.n_rows = 3
        self.n_rows = 2
        fig, axs = plt.subplots(nrows=self.n_rows, ncols=1, constrained_layout=True)
        fig.suptitle('Memory')
        for ax in axs:
            ax.axis('off')

        # Add Subfigures per Subplots
        gridspec = axs[0].get_subplotspec().get_gridspec()
        subfigs = [fig.add_subfigure(gs) for gs in gridspec]

        fig_titles = ['ST MEMORY', 'LT_MEMORY', 'Image_w_mask']

        for row, subfig in enumerate(subfigs):
            subfig.suptitle(fig_titles[row])

            # create 1x3 subplots per subfig
            # ST-Memory
            if 0 == row:
                axs = subfig.subplots(nrows=1, ncols=self.size_ST_Mem)
                for col, ax in enumerate(axs):
                    img = self.ST_memory_img[col] # load image
                    ax.imshow(img)
                    ax.set_title(f'# {self.ST_memory_idx[col]}')
                    ax.axis('off')

            # LT-Memory
            elif 1 == row:
                axs = subfig.subplots(nrows=1, ncols=self.size_LT_Mem)
                for col, ax in enumerate(axs):
                    img = self.LT_memory_img[col]
                    ax.imshow(img)
                    ax.set_title(f'# {self.LT_memory_idx[col]}')
                    ax.axis('off')

            # elif 2 == row:
            #     ax = subfig.subplots(nrows=1, ncols=1)
            #     ax.imshow(img_w_mask)
            #     ax.axis('off')

    def plt_save_Memory_plot(self, path_name_2_save_2:str, idx:int):
        folder_2_save_plots = os.path.join(path_name_2_save_2, self.path.split('/')[-1])
        # check if folder exists
        if not os.path.exists(folder_2_save_plots):
            os.makedirs(folder_2_save_plots)

        plt.savefig(os.path.join(folder_2_save_plots ,f'{idx}.png'))

    def show(self):
        plt.show()

########################################################################################################################
# UNIT TESTING
if '__main__' == __name__:
    ST_lim = 10
    LT_lim = 5
    plotting_instance = ST_LT_plotting_functions(ST_lim,LT_lim)

    path_2_images = 'Image_folder_for_test'
    plotting_instance.set_path_for_finding_the_images(path_2_images)

    st_indexes = []
    lt_indexes = []
    for idx in range(0,11):
        st_indexes.append(idx)
        lt_indexes.append(idx)

        plotting_instance.load_relevant_images(st_indexes.copy()[:ST_lim],lt_indexes.copy()[:LT_lim])
        start = time.time()
        img_with_mask = np.zeros([350,200,3])
        plotting_instance.plt_ST_and_LT_memory(img_with_mask)
        ic(time.time() - start)

        # plotting_instance.show()
        plotting_instance.plt_save_Memory_plot('READ_Mem',idx)
