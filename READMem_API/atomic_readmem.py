# Atomic class for the short- and long-term memory classes

# by Stéphane Vujasinovic

import numpy as np
import torch
import scipy

import lovely_tensors as lt
lt.monkey_patch()


def create_permutation_matrix(square_matrix_Tensor):
    square_matrix = square_matrix_Tensor.clone().detach().cpu().numpy()
    permutation_matrix = torch.zeros([*square_matrix_Tensor.shape])
    idx, jdx = scipy.optimize.linear_sum_assignment(square_matrix, maximize=True)
    permutation_matrix[idx,jdx] = 1.0

    return permutation_matrix

def _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, key:torch.Tensor, memory_keys:torch.Tensor,
                                                             affinity_matrices:torch.Tensor,
                                                             use_affinity:bool, debug_flag = False):
    similarites_list = []
    NBR_OF_OBJECTS, CHANNEL_SIZE = [*memory_keys.shape[:2]]
    SIZE = np.multiply(*memory_keys.shape[-2:])
    query_frame = key[:, :, 0]

    cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    dot_poduct = lambda x,y: torch.matmul(x.squeeze(),y.squeeze()) # both elements are flat vectors

    # This is legacy, from when I wanted to optimize the matrix operation
    # use_proper_permutation_matrix = True
    use_proper_permutation_matrix = False


    Matrix_view_of_CUR_frame = query_frame.view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()

    for frame_idx in range(0, memory_keys.shape[2]):
        Matrix_view_of_MEM_frame = memory_keys[:, :, frame_idx].view(NBR_OF_OBJECTS, CHANNEL_SIZE, SIZE).clone()
        # Matrix_view_of_MEM_frame_OG = Matrix_view_of_MEM_frame.clone()

        if use_affinity:
            affinity_matrix = affinity_matrices[frame_idx].clone()
            if not use_proper_permutation_matrix:
                dimension_for_sparse_axis = 0 # For QUERY # yields better results
                # dimension_for_sparse_axis = 1 # For MEMORY #

                values, indices = torch.topk(affinity_matrix, k=1,
                                             dim=dimension_for_sparse_axis)  # Extract the best score in the affinity along the query dimension

                sparse_affinity_matrices = torch.ones(
                    [*indices.shape]).cuda()  # Set ones where the best score along the query position.

                affinity_matrix = affinity_matrix.zero_().scatter_(dimension_for_sparse_axis, indices, sparse_affinity_matrices)

                if 0 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix
                elif 1 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix.T
                    # Matrix_view_of_CUR_frame = Matrix_view_of_CUR_frame @ affinity_matrix.T

            else:
                affinity_matrix = create_permutation_matrix(affinity_matrix).cuda()

                dimension_for_sparse_axis = 0 # For QUERY # yields better results
                if 0 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix
                elif 1 == dimension_for_sparse_axis:
                    Matrix_view_of_MEM_frame = Matrix_view_of_MEM_frame @ affinity_matrix.T


        if use_cosine_sim:
            similarities = [cos_similarity(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]
        else:
            similarities = [dot_poduct(Matrix_view_of_MEM_frame[obj_idx].flatten().unsqueeze(axis=0),
                                       Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                            in range(0, NBR_OF_OBJECTS)]


        # Testing if rounding the similarity gives me better results
        similarities = np.round(similarities, 5)


        similarites_list.append(np.array(similarities).mean())

    # Similarity with the remaining current frame to add at the end of the gram matrix
    if use_cosine_sim:
        similarities_current = [cos_similarity(Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0),
                                               Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for obj_idx
                                in range(0, NBR_OF_OBJECTS)]
    else:
        similarities_current = [dot_poduct(Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0),
                                           Matrix_view_of_CUR_frame[obj_idx].flatten().unsqueeze(axis=0)).item() for
                                obj_idx
                                in range(0, NBR_OF_OBJECTS)]
    # ic(similarities_current)
    similarities = np.round(similarities_current, 5)
    similarites_list.append(np.array(similarities).mean())

    return similarites_list


class AtomicREADMem:
    def __init__(self):
        self.reset_Mem()


    def reset_Mem(self):
        self.frame_indexes_list = []
        self.memory_keys   = None
        self.memory_values = None
        self.gram_det = None
        self.gram_matrix = None


    def similarity_annotated_frame(self, use_cosine_sim, memory_key_features_of_ANNOTATED_FRAME):
        h,w = memory_key_features_of_ANNOTATED_FRAME.shape[-2:]
        affinity_matrix = torch.diag(torch.ones(h*w))

        return _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, memory_key_features_of_ANNOTATED_FRAME, memory_key_features_of_ANNOTATED_FRAME, affinity_matrix,
                                                                        use_affinity=False, use_tukey=False, tukey_alpha = 1.0)


    @staticmethod
    def _compute_det(matrix, denominateur):
        return np.absolute(np.linalg.det(matrix/denominateur))
        # return torch.linalg.det(A)     If using torch for det, the use batch and not hust n vy n matrix, is going to ba faster


    @staticmethod
    def _compute_similarities(use_cosine_sim, key:torch.Tensor, memory_keys:torch.Tensor,
                              affinity_matrices:torch.Tensor,
                              use_affinity_flag=False) -> list:

        return _compute_similarity_with_memory_frames_and_current_frame(use_cosine_sim, key, memory_keys, affinity_matrices,
                                                                        use_affinity_flag)



    @staticmethod
    def _update_gram_matrix(gram_matrix:np.array, similarities:list) -> np.array: # version_for_dot_product # Meilleur version et devrait marcher avec cosine similarity si dans cosine similarity je fait aussi le calcule de la similarité avec le current frame
        gram_matrix = np.concatenate((gram_matrix, np.array([similarities[:-1]]).T), axis=1)
        gram_matrix = np.concatenate((gram_matrix, np.array([similarities])), axis=0)

        return gram_matrix


    @staticmethod
    def _delete_col_N_row_from_Gram_matrix(gram_matrix:np.array, idx_out: int) -> np.array:
        gram_matrix = np.delete(gram_matrix, idx_out, 0)
        gram_matrix = np.delete(gram_matrix, idx_out, 1)
        return gram_matrix


    @property
    def read_Mem(self):
        '''
        return/read/get memory
        '''
        return self.frame_indexes_list, self.memory_keys, self.memory_values


    def set_affinity_matrices(self, affinity_matrices):
        self.affinity_matrices = affinity_matrices


    def _compute_gram_det(self):
        denominateur = self.gram_matrix[0,0]
        self.gram_det = self._compute_det(self.gram_matrix,denominateur)


    def _get_gram_determinant(self) -> np.array:
        # ic('self.gram_det')
        # ic(self.gram_det)
        return self.gram_det

