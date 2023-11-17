"""
This file can handle DAVIS 2016/2017 evaluation.

modified by Stéphane Vujasinovic
"""
from MiVOS.model.propagation.prop_net import PropagationNetwork
from model.aggregate import aggregate_wbg
from util.tensor_util import pad_divide_by

from READMem_API.readmem import READMem

class InferenceCore:
    def __init__(self, prop_net: PropagationNetwork, num_objects, mem_config:str, debugging_flag=False, record_det_flag=False):
        self.prop_net = prop_net
        self.device = 'cuda'
        self.k = num_objects

        # Initialize READMem_API attributes
        self.readmem = READMem(mem_config, debugging_flag=True)
        self.readmem.nbr_of_objects_working_with = num_objects

        # Initialize flags for helping me debugging READMem
        self.debugging_flag = debugging_flag
        self.record_det_flag = record_det_flag


    def get_path_2_image_folder(self, img_folder: str):
        self.path_2_image_folder = img_folder


    def unpad(self, data, pad):
        if len(data.shape) == 4:
            if pad[2] + pad[3] > 0:
                data = data[:, :, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                data = data[:, :, :, pad[0]:-pad[1]]
        elif len(data.shape) == 3:
            if pad[2] + pad[3] > 0:
                data = data[:, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                data = data[:, :, pad[0]:-pad[1]]
        else:
            raise NotImplementedError
        return data


    def _get_query_kv_buffered(self, image):
        # not actually buffered
        return self.prop_net.get_query_values(image.cuda())


    def _set_image(self, sequence_length, OG_image):
        # True dimensions
        OG_image = OG_image.unsqueeze(dim=0).cuda()
        # self.t = sequence_length
        self.image, self.pad = pad_divide_by(OG_image, 16)


    def set_annotated_frame(self, idx, sequence_length, image, anno_mask):
        self.readmem.reset_readmem()    # Reset the ST and LT memories

        self._set_image(sequence_length, image)
        self.annotated_image = self.image.clone()

        anno_mask = anno_mask.unsqueeze(dim=1)
        mask, _ = pad_divide_by(anno_mask.cuda(), 16)
        self.prob = aggregate_wbg(mask, keep_bg=True)

        # KV pair for the interacting frame
        anno_key_k, anno_key_v = self.prop_net.memorize(self.image.cuda(), self.prob[1:].cuda())
        self.readmem.update_external_memory_with_readmem(idx, anno_key_k, anno_key_v)

        return self.unpad(self.prob,self.pad)


    def _adapt_img(self,OG_image):
        return pad_divide_by(OG_image.unsqueeze(dim=0).cuda(), 16)


    def step(self, idx, image):
        # Extract the key and values of the current frame
        img, pad = self._adapt_img(image)
        ori_img = img.clone()
        query = self._get_query_kv_buffered(img)

        # Extract the deep representation
        readmem_idx_list, readmem_keys, readmem_values = self.readmem.get_readmem_memory()

        # Infer the segmentation mask based on the deep representation stored in the external memory
        outmask = self.prop_net.segment_with_query(readmem_keys, readmem_values, *query)
        outmask = aggregate_wbg(outmask, keep_bg=True)
        self.prob = outmask

        prev_key, prev_value = self.prop_net.memorize(ori_img, outmask[1:])

        # Update the memory
        self.readmem.set_affinity_matrices(self.prop_net.get_affinity())
        self.readmem.update_external_memory_with_readmem(idx, prev_key, prev_value)

        return self.unpad(self.prob, pad)


    @property
    def return_lt_det(self):
        return self.readmem.LT_gram_det.copy()

    @property
    def ST_N_LT_Memories(self):
        return self.readmem.ST_Memory_indexes.copy(), self.readmem.LT_Memory_indexes.copy()

    @ST_N_LT_Memories.setter
    def ST_N_LT_Memories(self, new_st_indexes, new_lt_indexes):
        self.readmem.ST_Memory_indexes = new_st_indexes
        self.readmem.LT_Memory_indexes = new_lt_indexes

    @property
    def get_size_of_ST_N_LT_memory(self):
        return self.readmem.get_size_of_ST_N_LT_memory
