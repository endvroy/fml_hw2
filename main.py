from svmutil import *


def calc_scale():
    y, x = svm_read_problem('abalone.train', return_scipy=True)
    scale_param = csr_find_scale_param(x, lower=0)
    return scale_param


def cross_validation(d, c, scale_param):
    perf = 0
    for i in range(10):
        y, x = svm_read_problem(f'abalone.train.{i}', return_scipy=True)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c}')
        y_val, x_val = svm_read_problem(f'abalone.val.{i}', return_scipy=True)
        scaled_x_val = csr_scale(x_val, scale_param)
        p_label, p_acc, p_val = svm_predict(y_val, scaled_x_val, model)
        perf += p_acc[0]
    avg_perf = perf / 10
    return avg_perf


def cross_test(d, c, scale_param):
    perf = 0
    for i in range(10):
        y, x = svm_read_problem(f'abalone.train.{i}', return_scipy=True)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c}')
        y_val, x_val = svm_read_problem(f'abalone.test', return_scipy=True)
        scaled_x_val = csr_scale(x_val, scale_param)
        p_label, p_acc, p_val = svm_predict(y_val, scaled_x_val, model)
        perf += p_acc[0]
    avg_perf = perf / 10
    return avg_perf


def cross_count(d, c, scale_param):
    cnt = 0
    for i in range(10):
        y, x = svm_read_problem(f'abalone.train.{i}', return_scipy=True)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c}')
        cnt = model.get_nr_sv()
    avg_cnt = cnt / 10
    return avg_cnt


def cross_margin(d, c, scale_param):
    cnt = 0
    for i in range(10):
        y, x = svm_read_problem(f'abalone.train.{i}', return_scipy=True)
        scaled_x = csr_scale(x, scale_param)
        model = svm_train(y, scaled_x, f'-t 1 -d {d} -c {c}')
        coefs = model.get_sv_coef()
        for coef in coefs:
            if abs(coef[0]) != c:
                cnt += 1
    avg_cnt = cnt / 10
    return avg_cnt


if __name__ == '__main__':
    log = []
    record = []
    scale_param = calc_scale()
    k = 12
    for d in range(1, 5):
        # for log_c in range(-k, k + 1):
        #     c = 2 ** log_c
        c = 8
        perf = cross_test(d, c, scale_param)
        log_str = f'd={d} c={c} perf={perf}'
        log.append(log_str)
        record.append((d, c, perf))
    print('\n'.join(log))
