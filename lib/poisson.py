import numpy as np
import sparseqr
import scipy.sparse as sparse
import cv2
import itertools


class PoissonOperator(object):

    def __init__(self, data, mask, depth_info=None, depth_weight=0.1):
        h, w = mask.shape

        self.index_1d = np.ones([h, w]) * (-1)

        self.data = data
        self.mask = mask
        self.window_shape = (3, 3)
        self.valid_index = np.where(self.mask.ravel() != 0)[0]
        self.valid_num = len(self.valid_index)
        self.index_1d.reshape(-1)[self.valid_index] = np.arange(self.valid_num)

        self.depth = np.zeros([h, w])

        self.f_4neighbor = lambda x: np.array([x[1, 1], x[1, 2], x[0, 1], x[2, 1], x[1, 0]])

        # add depth_info and depth_weight for depth fusion
        self.depth_A = None
        self.depth_b = None
        if depth_info is not None:
            self.depth_A, self.depth_b = self.add_depth_info(depth_info, depth_weight)

    def add_depth_info(self, depth, w):
        rows, cols = depth.shape
        r = 0
        ind = 0
        col = []
        b = []
        variable_num = int(np.sum(self.mask))
        for i in range(rows):
            for j in range(cols):
                if self.mask[i, j]:
                    ind += 1
                if depth[i, j]:
                    r += 1
                    col.append(ind)
                    b.append(w*depth[i][j])

        data = np.array([w for i in range(r)])
        row = np.array([i for i in range(r)])
        col = np.array(col)

        A = sparse.coo_matrix((data, (row, col)), shape=(r, variable_num))
        b = np.array(b)
        return A, b

    def build_patch_for_poisson(self, mask_patch, data_patch, position_patch, weight=1):
        """
        get the cols and val for sparse matrix in this single patch
        :param mask_patch: 3*3 with weight
        :param data_patch: 3*3*d d is the dimension of the data, in normal case, we only need to input [p, q] 2d data
        :param position_patch: 3*3*1 the 1D patch position in the global image coordinate in 1d
        :param weight: the weight for this rows, which determine how important of this row
        :return: [colidx, colvals, bvals] colidx and colvals in 1d array with the same length, bval is a scaler
        """

        mask_used = self.f_4neighbor(mask_patch)
        data_used = self.f_4neighbor(data_patch)
        position_used = self.f_4neighbor(position_patch)

        colidx = []
        colvals = []
        bvals = 0

        if mask_used[1] == 1:
            D_ct = - (data_used[0] + data_used[1])[0] / 2 # the val between center to top
            colidx.append(position_used[1])
            colvals.append(1)
            bvals += D_ct
        if mask_used[2] == 1:
            D_cl = - (data_used[0] + data_used[2])[1] / 2# the val between center to left
            colidx.append(position_used[2])
            colvals.append(1)
            bvals += D_cl
        if mask_used[3] == 1:
            D_cr = (data_used[0] + data_used[3])[1] / 2 # the val between center to right
            colidx.append(position_used[3])
            colvals.append(1)
            bvals += D_cr
        if mask_used[4] == 1:
            D_cb = (data_used[0] + data_used[4])[0] / 2 # the val between center to bottom
            colidx.append(position_used[4])
            colvals.append(1)
            bvals += D_cb


        colidx.append(position_used[0])
        colvals.append(- np.sum(np.array(colvals)))

        return [colidx, colvals, bvals]

    def get_patches(self):
        # step 1: padding the data
        mask_pad = cv2.copyMakeBorder(self.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        from sklearn.feature_extraction.image import extract_patches_2d
        self.mask_patches = extract_patches_2d(mask_pad, self.window_shape)

        index_1d_pad = cv2.copyMakeBorder(self.index_1d, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=-1)
        self.index_1d_patches = extract_patches_2d(index_1d_pad, self.window_shape)
        data_pad = cv2.copyMakeBorder(self.data, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        self.data_patches = extract_patches_2d(data_pad, self.window_shape)

    def run(self):
        self.get_patches()
        self.poisson_b = []
        cols_all = []
        vals_all = []
        rows_all = []

        row_global = 0
        for i in self.valid_index:
            [colidx, colvals, bvals] = self.build_patch_for_poisson(self.mask_patches[i], self.data_patches[i], self.index_1d_patches[i])
            self.poisson_b.append(bvals)
            cols_all.append(colidx)
            vals_all.append(colvals)
            rows_all.append(np.ones_like(colidx) * row_global)
            row_global += 1

        rows_all_flat = list(itertools.chain.from_iterable(rows_all))
        cols_all_flat = list(itertools.chain.from_iterable(cols_all))
        vals_all_flat = list(itertools.chain.from_iterable(vals_all))

        self.poisson_A = sparse.coo_matrix((vals_all_flat, (rows_all_flat, cols_all_flat)), shape=(row_global, self.valid_num))
        self.poisson_b = np.array(self.poisson_b)

        # depth fusion
        if self.depth_A is not None:
            self.poisson_A = sparse.vstack((self.poisson_A,self.depth_A))
            self.poisson_b = np.hstack((self.poisson_b, self.depth_b))

        depth = sparseqr.solve(self.poisson_A, self.poisson_b, tolerance = 0)
        self.depth.reshape(-1)[self.valid_index] = depth
        return self.depth