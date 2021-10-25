
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc
import csv
import numpy as np
import plot
import random
import math
import statistics
from scipy.optimize import linprog


nu = 0.01
gamma_list = ['auto', 'scale']


def weighted_by_frequency(feature_vector_list):
    feature_vector_list_n = []
    for feature_vector in feature_vector_list:
        # convert string to int
        feature_vector = list(map(int, feature_vector))
        # weighted by term frequency
        K = sum(feature_vector)
        feature_vector_n = [l/K for l in feature_vector]
        feature_vector_list_n.append(feature_vector_n)

    return feature_vector_list_n


def separate_ss(input_list):
    """The input is a nested list, the function removes last element of each inner list; used for segment the segment
    sequence and the original feature vector
    """
    fv_list = []
    sequence_list = []
    for fv in input_list:
        fv_r = fv[:-1]
        segment_sequence = fv[-1]
        fv_list.append(fv_r)
        sequence_list.append(segment_sequence)
    fv_list = np.array(fv_list)
    sequence_list = np.array(sequence_list)
    # print("seprate_ss: ", fv_list.shape)
    return fv_list, sequence_list


def oc_svm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para=0.001,
           gamma_para='scale'):

    """ This function train an oc-svm classifier and compute FPR and TPR with default threshold;
    The trace segment is abnormal (attack) if prediction value == -1, the trace segment is normal
    if prediction value = 1

    INPUT: training_set, testing_set_normal, testing_set_attack are nested list of feature vectors
    dr: whether this function is applied after dimension reduction or not, the default value is not."""
    clf = OneClassSVM(nu=nu_para, kernel=kernel, gamma=gamma_para)
    clf.fit(training_set)
    if len(testing_set_normal) != 0:
        y_pred_test_normal = clf.predict(testing_set_normal)
        index = np.where(y_pred_test_normal == -1)
        n_error_test_normal = y_pred_test_normal[y_pred_test_normal == -1].size
        n_correct_test_normal = y_pred_test_normal[y_pred_test_normal == 1].size
        FP_rate = n_error_test_normal / len(testing_set_normal)

    else:
        FP_rate = -999

    if len(testing_set_attack) != 0:
        y_pred_test_attack = clf.predict(testing_set_attack)
        n_correct_test_attack = y_pred_test_attack[y_pred_test_attack == -1].size
        TP_rate = n_correct_test_attack / len(testing_set_attack)

    else:
        TP_rate = -999

    acc = (n_correct_test_attack+ n_correct_test_normal)/(len(testing_set_attack)+ len(testing_set_normal))

    return FP_rate, TP_rate, acc


def adver_samples_rank(training_set, testing_set_attack, kernel, nu_para,
           gamma_para='scale'):

    """Rank the adversary samples according to their distances to the decision boundary, return a list of indexes"""

    clf = OneClassSVM(nu=nu_para, kernel=kernel, gamma=gamma_para)
    clf.fit(training_set)

    distance_score = clf.decision_function(testing_set_attack)
    # print("distance_score: ", distance_score)

    ss_list = list(range(len(testing_set_attack)))
    index_distance_dict = dict(zip(ss_list, distance_score))
    # sort the dictionary from furthest to closest
    sorted_ss_distance_dict = {k: v for k, v in sorted(index_distance_dict.items(), key=lambda item: item[1])}
    adver_sample_index_list = list(sorted_ss_distance_dict.keys())

    return adver_sample_index_list


def random_selection(input_dataset, output_len, r_seed=10):
    """This function randomly select a number of samples from a dataset"""

    if r_seed != 0: # with fix seed
        input_dataset_index = list(range(len(input_dataset)))
        random.Random(r_seed).shuffle(input_dataset_index)
        output_dataset_index = input_dataset_index[: output_len]
        output_dataset = input_dataset[output_dataset_index]
    return output_dataset


def print_tuple_size(dataset, dataset_name="Shape: "):
    print(dataset_name, len(dataset), len(dataset[0]))


def alfa_solveQP(dataset_u, y_list, q, kernel, nu, gamma='scale'):
    eps = [0]*len(dataset_u) # hinge loss for the new classifier
    # construct the contaminated dataset
    dataset_a = []
    for i in range(len(dataset_u)):
        if q[i] != 0:
            dataset_a.append(dataset_u[i])

    dataset_a = np.array(dataset_a)
    new_classifier = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    new_classifier.fit(dataset_a)

    for i in range(len(dataset_u)):
        sample = dataset_u[i]
        normal_predicted = new_classifier.score_samples(sample.reshape(1, -1))
        eps[i] = max(0, 1 - y_list[i]*normal_predicted)

    return new_classifier, eps


def alfa_solveLP(training_normal, training_attack, dataset_u, eps, psi, C):
    """C =>Number of injected malicious samples, calculated with portions"""
    #TODO change to object-oriented coding style, using alfa as an object
    # func_coeff = [0] * len(training_attack)
    q_0 = [1]*len(training_normal)
    func_coeff = []
    for i in range(len(training_normal), len(dataset_u)):
        tmp = eps[i] - psi[i]
        func_coeff.append(tmp)
    # print(len(fun_coeff)==len(training_attack))
    # constraints
    a_eq = []
    b_eq = []
    tmp = [1]*len(training_attack)
    a_eq.append(tmp)
    b_eq.append(C)
    Q_bound = tuple([(0, 1)] * len(training_attack))
    q_1 = linprog(func_coeff, A_ub=a_eq, b_ub=b_eq, bounds=Q_bound,
                options={"disp": False, "maxiter": 30000}).x
    q = np.concatenate((np.array(q_0), q_1))
    return q

    

def ALFA(training_normal, training_attack, kernel, nu, C, gamma='scale'):
    """The normal dataset for training, training_attack --> the adversarial training"""
    # Initialize data
    normal_classifier = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    normal_classifier.fit(training_normal)

    dataset_u = np.concatenate((training_normal, training_attack))
    # print(training_normal.shape, training_attack.shape, dataset_uniform.shape)
    psi = [0] * len(dataset_u) # hinge loss for normal classifier
    eps = [0] * len(dataset_u)
    q_normal = [1]*len(training_normal)
    q_a = [0]*len(training_attack)
    q = q_normal + q_a
    y_list = [1]*len(training_normal) + [-1]*len(training_attack)

    # calculate psi list
    for i in range(len(dataset_u)):
        sample = dataset_u[i]
        normal_predicted = normal_classifier.score_samples(sample.reshape(1, -1))
        psi[i] = max(0, 1 - y_list[i]*normal_predicted)

    maxIter = 2
    curIter = 1
    while curIter <= maxIter:
        q = alfa_solveLP(training_normal, training_attack, dataset_u, eps, psi, C)
        new_classifier, eps = alfa_solveQP(dataset_u, y_list, q, kernel, nu)
        curIter += 1

    # print(q[:len(training_normal)])
    q_output = q[len(training_normal):]


    q_ind = np.argpartition(q_output, -C)[-C:]
    # print(q_output)
    # print(q_output[q_ind])
    training_attack_selected = training_attack[q_ind]
    return training_attack_selected


def label_flipping(dataset_list_normal, testing_set_attack, kernel, nu, op_sq):
    """This function contaminates the training datasets with various label flipping strategies and compute the
    performance degradation.
    op_sq: 0 ==> benchmark
           1 ==> random label flipping attack;
           2 ==> Furthest-first flip
           3 ==> Nearest-first flip
           4 ==> ALFA

    """


    # TODO: change the 4 dataset names and make them consistent to the
    # TODO: auto generate resulst that serves as the input

    if len(dataset_list_normal) == 0:
        raise ValueError("The input training data is empty!")

    """Separate the feature vectors and the segment sequences"""
    dataset_normal, dataset_normal_ss = separate_ss(dataset_list_normal)
    dataset_attack, dataset_attack_ss = separate_ss(testing_set_attack)

    """Split normal data into training (80) and test dataset (20)"""
    r_seed = 10
    dataset_normal_index = list(range(len(dataset_normal)))
    random.Random(r_seed).shuffle(dataset_normal_index)
    training_normal_len = math.floor(len(dataset_normal_index)*0.8)

    training_normal_index = dataset_normal_index[: training_normal_len]
    testing_normal_index =dataset_normal_index[training_normal_len:]

    training_normal = dataset_normal[training_normal_index]
    testing_normal = dataset_normal[testing_normal_index]

    """Split the attack data into test and tainted dataset"""
    r_seed = 10
    dataset_attack_index = list(range(len(dataset_attack)))
    random.Random(r_seed).shuffle(dataset_attack_index)
    #TODO IMPORTANT FIX THE SCENARIOS THAT len(dataset) is smaller than len(testing_normal)
    training_attack_len = len(dataset_attack)-len(testing_normal)
    # print(training_attack_len)

    training_attack_index = dataset_attack_index[: training_attack_len]
    testing_attack_index = dataset_attack_index[training_attack_len:]

    training_attack = dataset_attack[training_attack_index]
    testing_attack = dataset_attack[testing_attack_index]

    portion_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    if op_sq == 0:
        """Training with benchmark dataset"""
        FPR, TPR, acc = oc_svm(training_normal, testing_normal, testing_attack, kernel, nu)
        return acc

    elif op_sq == 1:
        """Uniform random flipping attacks"""

        portion_dict_acc_mean = {}
        portion_dict_acc_std = {}
        for portion in portion_list:
            num_adver_samples = math.ceil(portion*len(training_normal))
            # print(num_adver_samples)
            # randomize the malicious samples
            r_seed_list = list(range(1, 10))
            acc_list = []
            tpr_list = []
            fpr_list = []
            for r_seed in r_seed_list:
                adver_samples = random_selection(training_attack, num_adver_samples, r_seed)
                # print_tuple_size(adver_samples, "adver_samples: ")
                training_tainted = np.concatenate((training_normal, adver_samples))
                FPR, TPR, acc = oc_svm(training_tainted, testing_normal, testing_attack, kernel, nu)
                acc_list.append(acc)
                tpr_list.append(TPR)
                fpr_list.append(FPR)

            acc_mean = sum(acc_list)/len(acc_list)
            acc_std = statistics.stdev(acc_list)
            portion_dict_acc_mean[portion] = acc_mean
            portion_dict_acc_std[portion] = acc_std

        # print("sq_1_mean: ", portion_dict_acc_mean)
        # print("sq_1_std: ", portion_dict_acc_std)
        return [portion_dict_acc_mean, portion_dict_acc_std]

    elif op_sq == 2:
        """Furthest-first flip"""
        # portion_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        portion_dict_acc = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(training_normal, training_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion*len(training_normal))
            output_adver_samples_index = adver_samples_index[:num_adver_samples]
            adver_samples = training_attack[output_adver_samples_index]
            training_tainted = np.concatenate((training_normal, adver_samples))
            FPR, TPR, acc = oc_svm(training_tainted, testing_normal, testing_attack, kernel, nu)
            portion_dict_acc[portion] = acc
        # print("sq_2: ", portion_dict_acc)
        return portion_dict_acc

    elif op_sq==3:# Nearst-first flip
        # portion_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        portion_dict_acc = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(training_normal, training_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion * len(training_normal))
            output_adver_samples_index = adver_samples_index[len(training_attack)-num_adver_samples:]
            adver_samples = training_attack[output_adver_samples_index]
            training_tainted = np.concatenate((training_normal, adver_samples))
            FPR, TPR, acc = oc_svm(training_tainted, testing_normal, testing_attack, kernel, nu)
            portion_dict_acc[portion] = acc
        return portion_dict_acc
        # print("sq_3: ", portion_dict_acc)
    elif op_sq == 4: # ALFA
        portion_dict_acc = {}
        for portion in portion_list:
            print("portion: ", portion)
            num_adver_samples = math.ceil(portion * len(training_normal))
            num_adver_samples = min(num_adver_samples, len(training_attack))
            adver_samples =ALFA(training_normal, training_attack, kernel, nu, num_adver_samples)
            training_tainted = np.concatenate((training_normal, adver_samples))
            FPR, TPR, acc = oc_svm(training_tainted, testing_normal, testing_attack, kernel, nu)
            portion_dict_acc[portion] = acc
        return portion_dict_acc



def attack_sq_loop(data_list_normal, data_list_attack, kernel, nu, ad_attack_sq_list):
    """With fixed nu value"""
    sq_acc_dict = {}
    for sq in ad_attack_sq_list:
        print("sq: ", sq)
        acc_dict = label_flipping(data_list_normal.copy(), data_list_attack.copy(),
                                                                   kernel, nu, sq)
        # print(sq, ": ", acc_dict)
        sq_acc_dict[sq] = acc_dict

    return sq_acc_dict



def main():
    pass




if __name__ == "__main__":
    main()