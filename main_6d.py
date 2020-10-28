from svmutil import *
import numpy as np
from scipy import sparse


def get_new_x_y(x, y, d):
    x = np.array(x.todense())
    x = x[:-3]
    new_y = y[:-3]
    k = x @ np.transpose(x)
    new_x = np.power(k, d) * new_y
    return new_x, new_y


def get_chunk(x, y, i):
    val_x = x[313 * i:313 * (i + 1)]
    train_x = np.delete(x, slice(313 * i, 313 * (i + 1)), 0)
    val_y = y[313 * i:313 * (i + 1)]
    train_y = np.delete(y, slice(313 * i, 313 * (i + 1)), 0)
    return sparse.csr_matrix(train_x), train_y, sparse.csr_matrix(val_x), val_y


def cross_validation(orig_x, orig_y, d, c):
    new_x, new_y = get_new_x_y(orig_x, orig_y, d)
    scale_param = csr_find_scale_param(sparse.csr_matrix(new_x))
    perf = 0
    for i in range(10):
        x, y, x_val, y_val = get_chunk(new_x, new_y, i)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c} -h 0')
        scaled_x_val = csr_scale(x_val, scale_param)
        p_label, p_acc, p_val = svm_predict(y_val, scaled_x_val, model)
        perf += p_acc[0]
    avg_perf = perf / 10
    return avg_perf


def cross_test(orig_x, orig_y, d, c):
    new_x, new_y = get_new_x_y(orig_x, orig_y, d)
    scale_param = csr_find_scale_param(sparse.csr_matrix(new_x))
    perf = 0
    for i in range(10):
        x, y, x_val, y_val = get_chunk(new_x, new_y, i)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c} -h 0')
        y_val, x_val = svm_read_problem(f'abalone.test', return_scipy=True)
        scaled_x_val = csr_scale(x_val, scale_param)
        p_label, p_acc, p_val = svm_predict(y_val, scaled_x_val, model)
        perf += p_acc[0]
    avg_perf = perf / 10
    return avg_perf


if __name__ == '__main__':
    y, x = svm_read_problem('abalone.train.shuffled', return_scipy=True)
    log = []
    record = []
    k = 5
    for d in range(1, 5):
        # for log_c in range(-k, k + 1):
        #     c = 2 ** log_c
        c = 1024
        perf = cross_validation(x, y, d, c)
        log_str = f'd={d} c={c} perf={perf}'
        log.append(log_str)
        record.append((d, c, perf))
    print('\n'.join(log))
