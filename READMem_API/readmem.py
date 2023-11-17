# Contains all aspects needed for the short- and long-term memories.

# by Stéphane Vujasinovic

import numpy as np
import torch
import sys
import yaml
from icecream import ic

from READMem_API.short_term_memory_handler import ShortTermMemory
from READMem_API.long_term_memory_handler import LongTermMemory
sys.path.append("..")


def get_path_2_image_folder(self, img_folder: str):
    self.path_2_image_folder = img_folder


class READMem:
    def __init__(self, path_to_yaml_path_for_configuring_readmem:str, debugging_flag=False):
        # Initialize
        self.init_ST_Memory_flag = True
        self.init_LT_Memory_flag = True

        # Config READMem_API
        ST_config, LT_config = self._read_readmem_config(path_to_yaml_path_for_configuring_readmem)

        # Initialize ST and LT modules for READMem_API
        self.short_term_readmem = ShortTermMemory(ST_config)
        self.long_term_readmem = LongTermMemory(LT_config)

        self.debugging_flag = debugging_flag


    @property
    def nbr_of_objects_working_with(self):
        return self.nbr_of_objects_in_sequence


    @nbr_of_objects_working_with.setter
    def nbr_of_objects_working_with(self, nbr_of_objects_in_sequence:int):
        self.nbr_of_objects_in_sequence = nbr_of_objects_in_sequence


    @property
    def LT_gram_det(self):
        self.long_term_readmem._compute_gram_det()
        return self.long_term_readmem._get_gram_determinant()


    def set_affinity_matrices(self, affinity_matrices:torch.Tensor):
        ic('sub_check1')
        if self.use_affinity_base:
            affinity_matrices = affinity_matrices[1]
            ic(affinity_matrices)
        else:
            affinity_matrices = affinity_matrices[0]
            ic(affinity_matrices)


        ic('sub_check2')
        affinity_matrices_view = affinity_matrices.view(-1,affinity_matrices.shape[-1],affinity_matrices.shape[-1])
        if not self.init_ST_Memory_flag:
            self.mask_st = np.isin(np.array(self.readmem_frames_indexes_list), np.array(self.ST_frame_indexes_list))

            ic('sub_check3')
            affinity_matrices_ST = affinity_matrices_view[self.mask_st]  # Filter affinity matrix relevant for the short-term memory
            ic('sub_check4')
            self.short_term_readmem.set_affinity_matrices(affinity_matrices_ST)

        if not self.init_LT_Memory_flag:
            self.mask_lt = np.isin(np.array(self.readmem_frames_indexes_list), np.array(self.LT_frame_indexes_list))

            ic('sub_check5')
            affinity_matrices_LT = affinity_matrices_view[self.mask_lt]  # Filter affinity matrix relevant for the long-term memory
            ic('sub_check6')
            self.long_term_readmem.set_affinity_matrices(affinity_matrices_LT)


    def _read_readmem_config(self, yaml_config_file:str):
        with open(yaml_config_file, 'r') as yaml_file:  # Read yaml file and load params
            params = yaml.full_load(yaml_file)
            ST_config = params['ST_Mem']
            LT_config = params['LT_Mem']
            self.use_LT_memory_flag = params['Use_LT_memory']
            self.use_affinity_base = params['use_affinity_base']  # Either to use W or F

        return ST_config, LT_config


    def reset_readmem(self):
        self.init_ST_Memory_flag = True
        self.init_LT_Memory_flag = True

        self.short_term_readmem.reset_Mem()
        self.long_term_readmem.reset_Mem()


    def update_external_memory_with_readmem(self, idx, key, value):
        # Update the short- and long-term memories
        if self.init_ST_Memory_flag:
            self.init_ST_Memory_flag = False
            self.short_term_readmem.update_Mem(idx, key, value)
        else:
            self.short_term_readmem.update_Mem(idx, key, value)

        if not self.use_LT_memory_flag: return  # Skip Long term memory

        if self.init_LT_Memory_flag:
            self.init_LT_Memory_flag = False
            self.long_term_readmem.update_Mem_with_annotated_frame(idx, key, value)
            self.long_term_readmem.initialize_Gram_matrix(idx, key, value)
        else:
            # Prepare the short-term memory
            ST_gamma_diversity_ST = self.short_term_readmem.compute_ST_gamma_diversity()

            # Update the long-term memory based on the knowledge of the short-term memory
            self.long_term_readmem.update_Mem(idx, key, value, ST_gamma_diversity_ST)


    def get_readmem_memory(self):
        self.readmem_frames_indexes_list = []

        # Return only the ST memory because u don't care about LT memory
        if not self.use_LT_memory_flag:
            self.ST_frame_indexes_list, _, _ = [*self.short_term_readmem.read_Mem]
            self.readmem_frames_indexes_list = self.ST_frame_indexes_list
            return [*self.short_term_readmem.read_Mem]

        # Combine the ST and LT memories
        self.ST_frame_indexes_list, ST_memory_keys, ST_memory_values = [*self.short_term_readmem.read_Mem]
        self.LT_frame_indexes_list, LT_memory_keys, LT_memory_values = [*self.long_term_readmem.read_Mem]


        # Filter duplicates and arrange the memory frames in ascending order
        ST_frame_indexes_list_array = np.array(self.ST_frame_indexes_list)
        LT_frame_indexes_list_array = np.array(self.LT_frame_indexes_list)
        _,m = np.unique(LT_frame_indexes_list_array, True)
        LT_frame_indexes_list_array = LT_frame_indexes_list_array[m]
        m = np.isin(ST_frame_indexes_list_array, LT_frame_indexes_list_array, invert = True)        # Need another mask for the LT fames indexes
        sort_w_idx = np.argsort(np.concatenate((LT_frame_indexes_list_array,ST_frame_indexes_list_array[m])),axis=0)

        # Create READMem memory
        self.readmem_frames_indexes_list = np.concatenate((LT_frame_indexes_list_array, ST_frame_indexes_list_array[m]), axis=0)[sort_w_idx].tolist()

        ic(self.readmem_frames_indexes_list)
        ic(self.ST_frame_indexes_list)
        ic(self.LT_frame_indexes_list)

        if [] == self.ST_frame_indexes_list: # When not using ST for the adj frame
            self.readmem_frames_indexes_list = self.LT_frame_indexes_list
            return self.readmem_frames_indexes_list, LT_memory_keys, LT_memory_values


        readmem_memory_keys = torch.concat((LT_memory_keys, ST_memory_keys[:, :, m]), dim=2)[:,:,sort_w_idx]
        readmem_memory_values = torch.concat((LT_memory_values, ST_memory_values[:, :, m]), dim=2)[:,:,sort_w_idx]

        if self.debugging_flag:
            print('---------------------------------------------------------')
            print(self.ST_frame_indexes_list)
            print(self.LT_frame_indexes_list)
            print(self.readmem_frames_indexes_list)
            print('---------------------------------------------------------')

        return self.readmem_frames_indexes_list, readmem_memory_keys, readmem_memory_values


    @property
    def get_size_of_ST_N_LT_memory(self):
        return self.short_term_readmem.max_size_of_memory, self.long_term_readmem.max_size_of_memory

    @property
    def ST_Memory_indexes(self):
        return self.short_term_readmem.read_Mem[0]

    @ST_Memory_indexes.setter
    def ST_Memory_indexes(self, new_indexes:list):
        self.short_term_readmem.read_Mem[0] = new_indexes

    @property
    def LT_Memory_indexes(self):
        return self.long_term_readmem.read_Mem[0]

    @LT_Memory_indexes.setter
    def LT_Memory_indexes(self, new_indexes:list):
        self.short_term_readmem.read_Mem[0] = new_indexes

    def is_ST_Mem_going_to_be_updated_next_iteration(self, idx):
        return self.short_term_readmem.is_ST_Mem_going_to_be_updated_next_iteration(idx)

    @property
    def are_ST_N_LT_Mems_full(self):
        return self.short_term_readmem.is_ST_Mem_full

    @property
    def access_ST_Memory_elements(self):
        return self.short_term_readmem.access_ST_Memory_elements


    def re_write_ST_Memory_elements(self, n_ST_list_idx, n_memory_keys, n_memory_values, n_gram_matrix):
        self.short_term_readmem.re_write_ST_Memory_elements(n_ST_list_idx, n_memory_keys, n_memory_values, n_gram_matrix)













