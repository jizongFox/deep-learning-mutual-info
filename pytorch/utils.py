import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from deepclustering.utils import class2one_hot

NUM_CORES = multiprocessing.cpu_count()
bins = np.linspace(-1, 1, 30)
interval_information_display = 30

def get_information(ws, x, label, epoch_num=-1):

    pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)
    params = calc_information_for_epoch(epoch_num - 1, ws[epoch_num - 1], unique_inverse_x, unique_inverse_y, label,
                                        b, b1, len(unique_a), pys, pxs, p_y_given_x, pys1)
    return params


# give probabilities to discretized values of the outputs of layers
def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])
    b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    unique_array, unique_indices, unique_inverse_x, unique_counts = \
        np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    unique_a = x[unique_indices]
    b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
    pxs = unique_counts / float(np.sum(unique_counts))
    p_y_given_x = []
    for i in range(0, len(unique_array)):
        indexs = unique_inverse_x == i
        py_x_current = np.mean(label[indexs, :], axis=0)
        p_y_given_x.append(py_x_current)
    p_y_given_x = np.array(p_y_given_x).T
    b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
    unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
        np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
    pys1 = unique_counts_y / float(np.sum(unique_counts_y))
    return pys, pys1, p_y_given_x, b1, b, unique_a, unique_inverse_x, unique_inverse_y, pxs


def calc_entropy_for_specipic_t(current_ts, px_i):
    """Calc entropy for specipic t"""
    b2 = np.ascontiguousarray(current_ts).view(
        np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_current_ts = unique_counts / float(sum(unique_counts))
    p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
    H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
    return H2X


def calc_information_for_epoch(iter_index, ws_iter_index, unique_inverse_x,
                               unique_inverse_y, label, b, b1,
                               len_unique_a, pys, pxs, py_x, pys1):
    params = np.array(
        [calc_information_sampling(ws_iter_index[i], bins, pys1, pxs, label, b, b1,
                                   len_unique_a, py_x, unique_inverse_x,
                                   unique_inverse_y)
         for i in range(len(ws_iter_index))])

    return params


def calc_information_sampling(data, bins, pys1, pxs, label, b, b1, len_unique_a, p_YgX, unique_inverse_x,
                              unique_inverse_y):
    bins = bins.astype(np.float32)
    num_of_bins = bins.shape[0]

    digitized = bins[np.digitize(np.squeeze(data.reshape(1, -1)), bins) - 1].reshape(len(data), -1)
    b2 = np.ascontiguousarray(digitized).view(
        np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))
    unique_array, unique_inverse_t, unique_counts = \
        np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
    p_ts = unique_counts / float(sum(unique_counts))
    PXs, PYs = np.asarray(pxs).T, np.asarray(pys1).T
    local_IXT, local_ITY = calc_information_from_mat(PXs, PYs, p_ts, digitized, unique_inverse_x, unique_inverse_y,
                                                     unique_array)
    params = {}
    params['local_IXT'] = local_IXT
    params['local_ITY'] = local_ITY
    return params


def calc_information_from_mat(px, py, ps2, data, unique_inverse_x, unique_inverse_y, unique_array):
    """Calculate the MI based on binning of the data"""
    H2 = -np.sum(ps2 * np.log2(ps2))
    H2X = calc_condtion_entropy(px, data, unique_inverse_x)
    H2Y = calc_condtion_entropy(py.T, data, unique_inverse_y)
    IY = H2 - H2Y
    IX = H2 - H2X
    return IX, IY


def calc_condtion_entropy(px, t_data, unique_inverse_x):
    # Condition entropy of t given x
    H2X_array = np.array(
        Parallel(n_jobs=NUM_CORES)(delayed(calc_entropy_for_specipic_t)(t_data[unique_inverse_x == i, :], px[i])
                                   for i in range(px.shape[0])))
    H2X = np.sum(H2X_array)
    return H2X


def calc_probs(t_index, unique_inverse, label, b, b1, len_unique_a):
    """Calculate the p(x|T) and p(y|T)"""
    indexs = unique_inverse == t_index
    p_y_ts = np.sum(label[indexs], axis=0) / label[indexs].shape[0]
    unique_array_internal, unique_counts_internal = \
        np.unique(b[indexs], return_index=False, return_inverse=False, return_counts=True)
    indexes_x = np.where(np.in1d(b1, b[indexs]))
    p_x_ts = np.zeros(len_unique_a)
    p_x_ts[indexes_x] = unique_counts_internal / float(sum(unique_counts_internal))
    return p_x_ts, p_y_ts
def extract_array(data, name):
    results = [[data[j,][name]] for j in range(data.shape[0])]
    return results

import matplotlib.pyplot as plt
# plot information Plane
def plot_info_plane(i, I_XT_array, I_TY_array):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    plt.clf()
    plt.scatter(I_XT_array, I_TY_array, color=color)
    plt.title("Information Plane after " + str(i) + " Epochs")
    plt.ylim([0, 1])
    plt.xlim([0, 12])
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    # plt.savefig("plots/final/snapshot" + str(i) + ".png")
    plt.show(block=False)
    plt.pause(0.001)
