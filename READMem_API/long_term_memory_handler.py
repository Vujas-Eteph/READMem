# Handles only the long-term memory aspect of READMem_API

# by Stéphane Vujasinovic

import numpy as np
import torch
from icecream import ic

from READMem_API.atomic_readmem import AtomicREADMem

class LongTermMemory(AtomicREADMem):
    def __init__(self, LT_config):
        AtomicREADMem.__init__(self)
        # Configure long-term memory
        self.max_size_of_memory = LT_config['memory_size']
        self.mem_freq = LT_config['sampling_interval']
        self.similarity_bound = LT_config['similarity_bound']
        self.method_LT_memory = LT_config['method']
        self.LT_init_method = LT_config['init_method']
        self.keep_annotated_frame_in_LT_memory = LT_config['keep_annotated_frame_in_LT_memory']
        self.Use_affinity = LT_config['use_affinity']
        self.use_cosine_sim = LT_config['use_cosine_sim']

        if 'annotated and last' == self.method_LT_memory or 'annotated and max' == self.method_LT_memory:
            self.zeta = LT_config['zeta']

        self.updated_LT_Mem_flag = False

    def set_init_method_for_gram(self, LT_init_method):
        self.LT_init_method = LT_init_method

    def update_Mem_with_annotated_frame(self, idx, key, value):
        # Annotated frame is always kept in the LT memory as it is the one with a guaranteed good mask
        self.frame_indexes_list.append(idx)
        self.memory_keys = key
        self.memory_values = value
        self.affinity_matrices = torch.Tensor(np.eye(key.shape[-2]*key.shape[-1])).to(device='cuda:0').unsqueeze(dim=0)
        self.updated_LT_Mem_flag = True


    def initialize_Gram_matrix(self, idx, key, value):
        if self.gram_matrix is None:
            self.gram_matrix = np.array([[self._compute_similarities(self.use_cosine_sim, key, key, None)[0]]])
        if 'Every_M' == self.LT_init_method: return
        # Initialize the gram matrix
        self.Current_memory_size = self.memory_keys.shape[2]
        while self.Current_memory_size < self.max_size_of_memory:
            self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices,
                                                                                                     self.Use_affinity, self.Use_tukey_window, self.tukey_alpha))

            self.memory_keys = torch.cat([self.memory_keys,key], dim=2)
            self.memory_values = torch.cat([self.memory_values,value], dim=2)
            self.frame_indexes_list.append(idx)

            self.affinity_matrices = torch.cat([self.affinity_matrices,
                                                torch.Tensor(np.eye(key.shape[-2]*key.shape[-1])).to(device='cuda:0').unsqueeze(dim=0)],
                                               dim=0)
            self.Current_memory_size = self.memory_keys.shape[2]


    def update_Mem(self, idx, key, value, ST_gamma_diversity_ST):
        # ic(self.gram_matrix/self.gram_matrix[0,0])
        self.updated_LT_Mem_flag = False
        # After condition 0 and 1 are valid, add the new frame in the memory until full.
        # If the LT memroy is at full capacity, then check if the diversity is enhanced by replacing one of the memory frames with the current frame
        _idx, _key, _value = idx, key, value
        if not self._condition_only_every_i_th_frame_to_consider(_idx): return
        ic('_idx:',_idx)

        # ic(self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST))
        if not self._condition_for_only_considering_similar_frames(_key, ST_gamma_diversity_ST): return

        self._update_LT_Mem_part_1(_idx, _key, _value)


    def _condition_only_every_i_th_frame_to_consider(self, idx):
        # Condition that the new frame is not a redundant information
        # return self.mem_freq <= idx - self.frame_indexes_list[-1] # dynamic sampling interval
        return 0 == (idx % self.mem_freq) # static sampling interval



    def _update_LT_Mem_part_1(self, idx, key, value):
        if 'Every_M' == self.LT_init_method:
            if not self._check_that_LT_memory_is_full(idx,key,value): return
        # Throw away the annotated frame guaranteed
        # if self.discard_annotated_frame(): return
        condition_valid, best_idx = self._condition_to_update_LT_Mem_only_if_diversity_is_enhanced(key)
        if not condition_valid: return
        self._update_LT_Mem_part_2(idx, best_idx, key, value)
        self.updated_LT_Mem_flag = True

    def _check_that_LT_memory_is_full(self, idx, key, value):
        # Ensure that the LT memory is full, if not full, then full it.
        Current_memory_size = self.memory_keys.shape[2]
        if Current_memory_size >= self.max_size_of_memory: return True  # if Memory is not complete then

        # Update the gram matrix
        self.gram_matrix = self._update_gram_matrix(self.gram_matrix, self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices))

        # Update LT Memory
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys, key], dim=2)
        self.memory_values = torch.cat([self.memory_values, value], dim=2)
        return False

    def discard_annotated_frame(self):  # Not used in the default READMem, but can be done
        if 0 not in self.frame_indexes_list: return False
        self.frame_indexes_list = self.frame_indexes_list[1:]
        self.memory_keys = self.memory_keys[:,:,1:]
        self.memory_values =self.memory_values[:,:,1:]
        self.gram_matrix = self.gram_matrix[1:,1:]

        return True


    def _update_LT_Mem_part_2(self, idx, best_idx, key, value):
        # Replace the element in the memory with the better version
        del self.frame_indexes_list[best_idx]
        self.frame_indexes_list.append(idx)
        self.memory_keys = torch.cat([self.memory_keys[:, :, :best_idx],
                                      self.memory_keys[:, :, best_idx + 1:],
                                      key], dim=2)
        self.memory_values = torch.cat([self.memory_values[:, :, :best_idx],
                                        self.memory_values[:, :, best_idx + 1:],
                                        value], dim=2)

    def _condition_for_only_considering_similar_frames(self, key, ST_gamma_diversity = 0):
        # Check with the similarity of the current frame and the ones in the LT memory.
        # If similarity too low, might be because too much background taken into account.
        values,indexes,counts = np.unique(np.array(self.frame_indexes_list),True,False,True)

        # Just the clone, copy the affinity of the annotated frame and give it to the other also! ATTENTION ! ONLY FOR THE INITIALIZATION !!
        for v, i, c in zip(values, indexes,counts):  # This only works at the initialization and is not a general solution. !!!!!
            if 1 == c: continue

            affinity_to_clone = self.affinity_matrices[i,:,:].clone() # Take the index 0 because its the annotated variant
            new_affinities = torch.zeros(c-1,*affinity_to_clone.shape).cuda()
            for j in range(0,c-1):
                new_affinities[j]+=affinity_to_clone

            self.affinity_matrices = torch.cat([new_affinities, self.affinity_matrices], dim=0) # Add at the beginning of the extract affinity matrices, as you want to add affiniies only for the annotation frame

        # Create temporary_affinity_matrix that matches the number of elements in the memory. Very important to be constistent
        # with the number of frames stored in the lt memory and the affinity matrices.

        similarities = self._compute_similarities(self.use_cosine_sim, key, self.memory_keys, self.affinity_matrices, self.Use_affinity)
        similarity_with_annotated_frame = similarities[0]

        # See if second condition is valid: That the similarity is good enough
        ic(similarity_with_annotated_frame)
        ic(self.similarity_bound)
        if 'static lower bound' == self.method_LT_memory:
            return similarity_with_annotated_frame/self.gram_matrix[0,0]> self.similarity_bound

        elif 'dynamic lower bound' == self.method_LT_memory:
            return similarity_with_annotated_frame/self.gram_matrix[0,0] > self.similarity_bound - ST_gamma_diversity

        elif 'ensemble lower bound' == self.method_LT_memory:
            return np.array(similarities).mean()/self.gram_matrix[0,0] > self.similarity_bound

        elif 'last' == self.method_LT_memory:
            return similarities[-1]/self.gram_matrix[0,0] > self.similarity_bound

        elif 'ensemble max' == self.method_LT_memory:
            return np.array(similarities[:-1]).max()/self.gram_matrix[0,0] > self.similarity_bound

        elif 'annotated and last' == self.method_LT_memory:
            return ((1 - self.zeta) * similarities[-1] + self.zeta * similarity_with_annotated_frame)/self.gram_matrix[0, 0] > self.similarity_bound

        elif 'annotated and max' == self.method_LT_memory:
            return ((1 - self.zeta) * np.array(similarities).max() + self.zeta * similarity_with_annotated_frame)/self.gram_matrix[0, 0] > self.similarity_bound



    def _condition_to_update_LT_Mem_only_if_diversity_is_enhanced(self, key):
        # Determine if the volume of the parallelotope gets bigger by including the next candidate frame in the long-term memory.
        self._compute_gram_det()

        starting_position = 1 if self.keep_annotated_frame_in_LT_memory else 0
        temporary_determinants_list = []
        temporary_gram_matrix_list = []

        denominateur = self.gram_matrix[0,0]

        for idx in range(starting_position, self.max_size_of_memory):
            temporary_mem = torch.cat((self.memory_keys[:,:,:idx], self.memory_keys[:,:,idx+1:]), dim=2)
            temporary_affinity = torch.cat((self.affinity_matrices[:idx,:,:],self.affinity_matrices[idx+1:,:,:]),dim=0)
            temporary_gram_matrix = self.gram_matrix.copy()

            temporary_similarities = self._compute_similarities(self.use_cosine_sim, key, temporary_mem, temporary_affinity, self.Use_affinity)
            temporary_gram_matrix = self._delete_col_N_row_from_Gram_matrix(temporary_gram_matrix, idx)
            temporary_gram_matrix = self._update_gram_matrix(temporary_gram_matrix, temporary_similarities)
            temporary_determinant = self._compute_det(temporary_gram_matrix,denominateur)

            temporary_gram_matrix_list.append(temporary_gram_matrix)
            temporary_determinants_list.append(temporary_determinant)

        # Check if determinant is bigger then the base determinant
        best_idx = np.argmax(temporary_determinants_list)
        best_temporary_det = temporary_determinants_list[best_idx]

        ic(best_temporary_det)
        ic(self.gram_det)

        if self.gram_det <= best_temporary_det:
            self.gram_matrix = temporary_gram_matrix_list[best_idx]
            best_idx += starting_position

            return True, best_idx
        else:
            return False, None


    @property
    def is_LT_Mem_updated(self):
        return self.updated_LT_Mem_flag

