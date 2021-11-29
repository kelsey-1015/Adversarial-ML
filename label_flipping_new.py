
from sklearn.svm import OneClassSVM
import numpy as np
import random
import math
import statistics
from scipy.optimize import linprog
from more_itertools import take



nu = 0.01
gamma_list = ['auto', 'scale']

# save the configuration of different experiment settings
configuration_dict = {"ADFA_LD_single": {"nr_train_normal": 300, "portion_train_test": 0.9}}


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
    return fv_list, sequence_list


def oc_svm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para,
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

    """Rank the adversary samples according to their distances to the decision boundary in descending order,
     return a list of indexes"""

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

    q_output = q[len(training_normal):]


    q_ind = np.argpartition(q_output, -C)[-C:]
    training_attack_selected = training_attack[q_ind]
    return training_attack_selected


def label_flipping_db_generation(train_normal, train_attack, op_sq, portion_list, kernel, nu=0.01):
    """This function generates the tainted training dataset with different attack strategies"""
    if op_sq == 2:
        """Furthest-first flip"""
        portion_tainted_db = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(train_normal, train_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion*len(train_normal))
            output_adver_samples_index = adver_samples_index[:num_adver_samples]
            adver_samples = train_attack[output_adver_samples_index]
            training_tainted = np.concatenate((train_normal, adver_samples))
            label_true = np.array([0]*len(train_normal) + [-1]*len(adver_samples))
            portion_tainted_db[portion] = [label_true, training_tainted]
        return portion_tainted_db

    elif op_sq == 3:
        # Nearst-first flip
        portion_tainted_db = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(train_normal, train_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion * len(train_normal))

            if num_adver_samples > len(train_attack):
                # raise ValueError("The input portion exceeds the maximum number!")
                print("The input portion {} exceeds the maximum number!".format(portion))
                break
            else:
                output_adver_samples_index = adver_samples_index[(len(train_attack)-num_adver_samples):]
                adver_samples = train_attack[output_adver_samples_index]
                training_tainted = np.concatenate((train_normal, adver_samples))
                label_true = np.array([0] * len(train_normal) + [-1] * len(adver_samples))
                portion_tainted_db[portion] = [label_true, training_tainted]
        return portion_tainted_db

    elif op_sq == 4:
        # ALFA
        portion_tainted_db = {}
        for portion in portion_list:
            num_adver_samples = math.ceil(portion * len(train_normal))
            num_adver_samples = min(num_adver_samples, len(train_attack))
            adver_samples = ALFA(train_normal, train_attack, kernel, nu, num_adver_samples)
            training_tainted = np.concatenate((train_normal, adver_samples))
            label_true = np.array([0] * len(train_normal) + [-1] * len(adver_samples))
            portion_tainted_db[portion] = [label_true, training_tainted]
        return portion_tainted_db


#



def label_flipping_san(san_dict, train_normal, train_attack, test_normal, test_attack,  op_sq, portion_list, kernel,
                       nu=0.01):
    """This function contaminates the training datasets with various label flipping strategies and compute the
    performance degradation.
    op_sq: 0 ==> benchmark
           1 ==> random label flipping attack;
           2 ==> Furthest-first flip
           3 ==> Nearest-first flip
           4 ==> ALFA
    INPUT: train_normal, train_attack, test_normal, test_attack are np arrays of preferred size

    """

    if op_sq == 0:
        """Training with benchmark dataset"""
        FPR, TPR, acc = oc_svm(train_normal, test_normal, test_attack, kernel, nu)
        return acc

    if op_sq == 2:
        """Furthest-first flip"""
        portion_dict_acc = {}
        portion_dict_acc_s = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(train_normal, train_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion*len(train_normal))
            output_adver_samples_index = adver_samples_index[:num_adver_samples]
            adver_samples = train_attack[output_adver_samples_index]
            training_tainted = np.concatenate((train_normal, adver_samples))
            FPR, TPR, acc = oc_svm(training_tainted, test_normal, test_attack, kernel, nu)
            san_dataset = san_dict[op_sq][portion]
            FPR_s, TPR_S, acc_s = oc_svm(san_dataset, test_normal, test_attack, kernel, nu)
            portion_dict_acc[portion] = acc
            portion_dict_acc_s[portion] = acc_s
        return portion_dict_acc, portion_dict_acc_s

    elif op_sq == 3:# Nearst-first flip
        portion_dict_acc = {}
        portion_dict_acc_s = {}
        # furthest to nearest
        adver_samples_index = adver_samples_rank(train_normal, train_attack, kernel, nu)
        for portion in portion_list:
            num_adver_samples = math.ceil(portion * len(train_normal))
            if num_adver_samples > len(train_attack):
                # raise ValueError("The input portion exceeds the maximum number!")
                print("The input portion {} exceeds the maximum number!".format(portion))
                break
            else:
                output_adver_samples_index = adver_samples_index[(len(train_attack)-num_adver_samples):]
                adver_samples = train_attack[output_adver_samples_index]
                training_tainted = np.concatenate((train_normal, adver_samples))
                FPR, TPR, acc = oc_svm(training_tainted, test_normal, test_attack, kernel, nu)
                san_dataset = san_dict[op_sq][portion]
                FPR_s, TPR_S, acc_s = oc_svm(san_dataset, test_normal, test_attack, kernel, nu)
                portion_dict_acc[portion] = acc
                portion_dict_acc_s[portion] = acc_s
        return portion_dict_acc, portion_dict_acc_s


    elif op_sq == 4: # ALFA
        portion_dict_acc = {}
        portion_dict_acc_s = {}
        for portion in portion_list:
            print("portion: ", portion)
            num_adver_samples = math.ceil(portion * len(train_normal))
            num_adver_samples = min(num_adver_samples, len(train_attack))
            adver_samples = ALFA(train_normal, train_attack, kernel, nu, num_adver_samples)
            training_tainted = np.concatenate((train_normal, adver_samples))
            FPR, TPR, acc = oc_svm(training_tainted, test_normal, test_attack, kernel, nu)
            san_dataset = san_dict[op_sq][portion]
            FPR_s, TPR_S, acc_s = oc_svm(san_dataset, test_normal, test_attack, kernel, nu)
            portion_dict_acc[portion] = acc
            portion_dict_acc_s[portion] = acc_s
        return portion_dict_acc, portion_dict_acc_s



def attack_sq_loop(train_normal, train_attack, test_normal, test_attack, kernel, nu, ad_attack_sq_list, portion_list):
    """With fixed nu value"""
    sq_acc_dict = {}
    for sq in ad_attack_sq_list:
        print("sq: ", sq)
        acc_dict = label_flipping(train_normal.copy(), train_attack.copy(), test_normal.copy(), test_attack.copy(),
                                                                   kernel, nu, sq, portion_list)
        sq_acc_dict[sq] = acc_dict

    return sq_acc_dict


def data_preprocessing_multiple_attack(file_tf_dict, attack_name_test, attack_name_train_list, nr_train_normal,
                                       portion_train_test):
    """This function tests the classifier with one attack and injected malicious points of the remaining attacks"""
    #TODO We all pickup the first N elements if there are more provided samples than the requrid samples.
    #TODO Some work need to be done to rankdomly pickup
    train_normal = []
    train_normal_dict = file_tf_dict["Training_Data_Master"]
    # construct train normal
    if nr_train_normal > len(train_normal_dict):
        raise ValueError
    else:
        for k in train_normal_dict.values():
            train_normal.append(list(k.values()))

    train_normal = train_normal[: nr_train_normal]
    train_normal = np.array(train_normal)

    # compute nr_test_normal based on the train_test_ratio, note we get the first n elments from the validation set
    nr_test_normal = math.ceil((1 - portion_train_test) * nr_train_normal)
    test_normal = []
    test_normal_dict = file_tf_dict["Validation_Data_Master"]
    # construct test normal
    index_t = 0
    for k in test_normal_dict.values():
        if index_t < nr_test_normal:
            test_normal.append(list(k.values()))
            index_t += 1
        else:
            break
    test_normal = np.array(test_normal)

    # Generate np arrays of samples for test attack
    test_attack = []
    test_attack_dict = file_tf_dict["Attack_Data_Master"][attack_name_test]
    nr_test_attack = len(test_attack_dict)
    # To construct a balanced dataset
    if nr_test_normal < nr_test_attack:
        nr_test_attack = nr_test_normal
    else:
        raise ValueError("The nr_attack_test is larger than the nr_attack!")

    for k in test_attack_dict.values():
        test_attack.append(list(k.values()))

    test_attack = test_attack[: nr_test_attack]
    test_attack = np.array(test_attack)

    # Generate np arrays of samples for train attacks (a list)
    train_attack = []
    for attack_name_train in attack_name_train_list:
        train_attack_dict = file_tf_dict["Attack_Data_Master"][attack_name_train]
        train_attack_single = []
        for k in train_attack_dict.values():
            train_attack_single.append(list(k.values()))
        train_attack = train_attack + train_attack_single
    train_attack = np.array(train_attack)

    # print(train_normal.shape)
    # print(test_normal.shape)
    # print(train_attack.shape)
    # print(test_attack.shape)


    return train_normal, train_attack, test_normal, test_attack


def data_preprocessing(file_tf_dict, attack_name, nr_train_normal, portion_train_test):
    train_normal = []
    train_normal_dict = file_tf_dict["Training_Data_Master"]

    # compute nr_test_normal based on the train_test_ratio
    nr_test_normal = math.ceil((1-portion_train_test)*nr_train_normal)
    test_normal = []
    test_normal_dict = file_tf_dict["Validation_Data_Master"]

    attack_dataset = []
    attack_dataset_dict = file_tf_dict["Attack_Data_Master"][attack_name]
    nr_attack_dataset = len(attack_dataset_dict)

    # construct equvalent dataset
    if nr_test_normal < nr_attack_dataset:
        nr_test_attack = nr_test_normal
    else:
        raise ValueError("The nr_attack_test is larger than the nr_attack!")

    nr_train_attack = nr_attack_dataset - nr_test_attack


    max_portion = nr_train_attack/nr_train_normal
    print("Max portion: ", max_portion)


    # construct train normal
    if nr_train_normal > len(train_normal_dict):
        raise ValueError
    else:
        for k in train_normal_dict.values():
            train_normal.append(list(k.values()))

        train_normal = train_normal[:nr_train_normal]
        train_normal = np.array(train_normal)

    # construct test normal
    index_t = 0
    for k in test_normal_dict.values():
        if index_t<nr_test_normal:
            test_normal.append(list(k.values()))
            index_t += 1
        else:
            break
    test_normal = np.array(test_normal)


    for k in attack_dataset_dict.values():
        attack_dataset.append(list(k.values()))

    test_attack = attack_dataset[: nr_test_attack]
    test_attack = np.array(test_attack)

    train_attack = attack_dataset[nr_test_attack:]
    train_attack = np.array(train_attack)

    return train_normal, train_attack, test_normal, test_attack



def main():
    data_preprocessing_multiple_attack()




if __name__ == "__main__":
    main()