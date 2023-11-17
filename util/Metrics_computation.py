# Class for handling Mask operations
#
# by Stéphane Vujasinovic

import numpy as np

class Handle_Metrics():
    def __init__(self):
        pass

    def compute_Jaccard_Index(self, ground_truth_masks, predicted_masks):
        '''
        Very simple, only works for Single Object of Interest (SOoI)
        :param GT_mask:
        :param Pred_mask:
        :return:
        '''
        if 0 == ground_truth_masks.sum():  # Means the OoI is occluded
            return None

        mask_diff = ground_truth_masks - predicted_masks
        mask_add = ground_truth_masks + predicted_masks
        TP_mask = mask_add == 2  # ou je trouve des 2
        FN_mask = mask_diff == 1  # ou je trouve des 1
        FP_mask = mask_diff == 255  # ou je trouve des -1/ 255 because uint8

        Jaccard_index = TP_mask.sum()/(FP_mask.sum()+FN_mask.sum()+TP_mask.sum())

        return Jaccard_index


#
# - TEST
if "__main__" == __name__ :
    GT_mask = np.array([[0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,1,1,0,0]], dtype=np.uint8)

    Pred_mask = np.array([[1,0,0,0,0,0,1],
                          [1,0,1,1,1,0,1],
                          [1,0,1,0,1,0,1],
                          [1,0,1,1,1,0,1],
                          [1,0,0,0,0,0,1]], dtype=np.uint8)
    # Pred_mask = GT_mask.copy()

    metric_handler = Handle_Metrics()
    res = metric_handler.compute_Jaccard_Index(GT_mask, Pred_mask)
    print(res)