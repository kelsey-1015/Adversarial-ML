import numpy as np
import plot as plt
from sklearn.cluster import DBSCAN
import math
from ADFA_LD_parser import convert_to_dict_total, unique_symbol_dict_generation, convert_to_dict_tf
import label_flipping_new as LFP
import trace_file_parser as tp
from constants import *
import json
from sklearn.decomposition import TruncatedSVD
import random
from sklearn.decomposition import PCA
import json

attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
ADFA_LD_attack_name = ["Add User", "Java Meterpreter", "Web Shell"]
"""1 ==> euc; 2 ==> cross entropy; 3: square ; sq: [eps, minP]"""
optimal_para_config = {1: [0.3, 100], 2: [0.81, 60], 3: [0.21, 100]}
# optimal_para_config = {1: [0.3, 100], 2: [0.81, 80], 3: [0.21, 100]}
# euclidean = {'eps': 0.3, 'minP': 100}
# cross_entropy ={'eps': 0.61, 'minP': 100}
# square = {'eps': 0.21, 'minP': 100}
dl4ld_dataset_split_para = {'couchdb': {'kernel': "linear", 'ratio': 0.9}, 'mongodb': {'kernel': "linear", 'ratio': 0.8}}
adfa_dataset_split_para = {'kernel': "rbf", 'ratio': 0.9}
nr_train_parameter_search = 300


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


def extract_feature_vector(rawtrace_file, feature_dict_file, Flag, segment_length, filter_flag, n_gram_length):
    """ parse raw trace and extracts feature vectors.,
    INPUT:Flag = 0 for tf, 1 for idf-tf, 2 for n-gram
    --> segment_length: how the raw tracefile is segmented
    --> filter_flag: whether specific syscalls are filtered out.
    OUTPUT: a nested list of feature vectors after normalization"""

    feature_dict = json.load(open(feature_dict_file))

    if Flag == 0:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict, segment_length,
                                                                     filter_flag)
        feature_vector_list = tp.normalization(feature_vector_list)

    if Flag == 1:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict, segment_length,
                                                                     filter_flag)
        feature_vector_list = tp.normalization(feature_vector_list)
        feature_vector_list = tp.df_idf(feature_vector_list, occurrence_dict, N)

    if Flag == 2:
        feature_vector_list = tp.parse_trace_ngram(rawtrace_file, feature_dict, segment_length, filter_flag,
                                                   n_gram_length)
        feature_vector_list = tp.normalization(feature_vector_list)

    # change into a numpy array for consistence
    feature_vector_list = np.array(feature_vector_list)
    return feature_vector_list

def dataset_concatenate(rawtrace_file_list, Flag, feature_dict_file, segment_length, filter_flag, n_gram_length):
    """This function combines data from multiple lists into an np array, col_num equals to the number of keys in
    the dist"""
    if isinstance(rawtrace_file_list, str): # if the input list is a string, then output directly
        dataset = extract_feature_vector(rawtrace_file_list, feature_dict_file, Flag, segment_length, filter_flag,
                                         n_gram_length)
        return dataset
    else:
        feature_dict = json.load(open(feature_dict_file))
        col_num = len(feature_dict)+1
        dataset_total = np.empty((0, col_num))
        for rawtrace_file in rawtrace_file_list:
            dataset = extract_feature_vector(rawtrace_file, feature_dict_file, Flag, segment_length, filter_flag,
                                             n_gram_length)
            dataset_total = np.concatenate((dataset_total, dataset))
        return dataset_total


def dl4ld_db_processing(app_name, train_test_ratio, feature_extraction="TF"):
    """The default random seed is 10"""
    n_gram_length = 3
    segment_length = 30000
    filter_flag = True
    r_seed = 10
    # train_test_ratio = 0.9

    """OUTPUT train_normal, train_attack, test_normal, test_attack"""
    feature_dict_file = FEATURE_DICT_FILE[feature_extraction]
    rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack']
    feature_extraction_index = FEATURE_VECTOR[feature_extraction]

    normal_dataset_total = dataset_concatenate(rawtrace_file_normal, feature_extraction_index, feature_dict_file,
                                               segment_length, filter_flag, n_gram_length)
    attack_dataset_total = extract_feature_vector(rawtrace_file_attack, feature_dict_file, feature_extraction_index,
                                                  segment_length, filter_flag, n_gram_length)

    """Separate the feature vectors and the segment sequences"""
    dataset_normal, dataset_normal_ss = separate_ss(normal_dataset_total)
    dataset_attack, dataset_attack_ss = separate_ss(attack_dataset_total)

    """Split normal data into training and test dataset according to the train_test_ratio parameter"""
    dataset_normal_index = list(range(len(dataset_normal)))
    random.Random(r_seed).shuffle(dataset_normal_index)
    training_normal_len = math.floor(len(dataset_normal_index) * train_test_ratio)

    train_normal_index = dataset_normal_index[: training_normal_len]
    test_normal_index = dataset_normal_index[training_normal_len:]

    train_normal = dataset_normal[train_normal_index]
    test_normal = dataset_normal[test_normal_index]

    """Split the attack data into test and tainted dataset"""
    dataset_attack_index = list(range(len(dataset_attack)))
    random.Random(r_seed).shuffle(dataset_attack_index)
    # contruct a balanced test dataset
    train_attack_len = len(dataset_attack) - len(test_normal)

    train_attack_index = dataset_attack_index[: train_attack_len]
    test_attack_index = dataset_attack_index[train_attack_len:]

    train_attack = dataset_attack[train_attack_index]
    test_attack = dataset_attack[test_attack_index]

    return train_normal, train_attack, test_normal, test_attack



def adfa_db_parser_total(attack_name, laplace_smoothing):
    """Convert the total dict to np array of tf"""

    output = convert_to_dict_total()
    feature_dict = unique_symbol_dict_generation(output)
    tf_dict_adfa = convert_to_dict_tf(feature_dict, laplace_smoothing)

    train_normal_dict = tf_dict_adfa["Training_Data_Master"]
    test_normal_dict = tf_dict_adfa["Validation_Data_Master"]
    attack_dataset_dict = tf_dict_adfa["Attack_Data_Master"][attack_name]

    train_normal = []
    for k in train_normal_dict.values():
        train_normal.append(list(k.values()))
    train_normal = np.array(train_normal)

    test_normal = []
    for k in test_normal_dict.values():
        test_normal.append(list(k.values()))
    test_normal = np.array(test_normal)

    attack_dataset = []
    for k in attack_dataset_dict.values():
        attack_dataset.append(list(k.values()))
    attack_dataset = np.array(attack_dataset)


    return train_normal, test_normal, attack_dataset


def adfa_db_preprocessing(attack_name, laplace_smoothing, nr_train_normal, portion_train_test):
    output = convert_to_dict_total()
    feature_dict = unique_symbol_dict_generation(output)
    tf_dict_adfa = convert_to_dict_tf(feature_dict, laplace_smoothing)
    train_normal, train_attack, test_normal, test_attack = LFP.data_preprocessing(tf_dict_adfa, attack_name,
                                                                                  nr_train_normal, portion_train_test)
    return train_normal, train_attack, test_normal, test_attack


def mydistance(x,y):
  return np.sum((x-y)**2)


def dime_reduction_pca(input_array, nr_dimension):
    pca = PCA(n_components=nr_dimension)
    pca.fit(input_array)
    output_array = pca.transform(input_array)
    return output_array


def dime_reduction_trunctedsvd(input_array, nr_dimension):
    svd = TruncatedSVD(n_components=nr_dimension)
    svd.fit(input_array)
    output_array = svd.transform(input_array)
    return output_array


def dbscan_dr(input_dataset, eps_i, min_s_i, distance_metric, dr_op, nr_dimension):

    if dr_op == 1: # Truncted SVD
        input_dataset = dime_reduction_trunctedsvd(input_dataset, nr_dimension)
    elif dr_op == 2: # pcs
        input_dataset = dime_reduction_pca(input_dataset, nr_dimension)

    if distance_metric == 1:
        db = DBSCAN(eps=eps_i, min_samples=min_s_i).fit(input_dataset)
        labels_pred = db.labels_
    elif distance_metric == 2:  # cross entropy, the input arrays are TFs with Laplace smoothing
        db = DBSCAN(eps=eps_i, min_samples=min_s_i, metric=cross_entropy).fit(input_dataset)
        labels_pred = db.labels_
    elif distance_metric == 3:  # square distance
        db = DBSCAN(eps=eps_i, min_samples=min_s_i, metric=mydistance).fit(input_dataset)
        labels_pred = db.labels_

    n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    n_noise = list(labels_pred).count(-1)

    return labels_pred, n_clusters, n_noise


def sanitization(original_dataset, labels):
    output_dataset = original_dataset[labels!=-1]
    return output_dataset



def cross_entropy(list1, list2):
    tmp_list = []
    for index in range(len(list1)):
        p = list1[index]
        q = list2[index]
        tmp = (p - q) * math.log(p / q, 2)
        tmp_list.append(tmp)
    c_entropy = sum(tmp_list)
    return c_entropy


def parameter_search(nr_parameter_search, distance_metric):
    """ This function help calculate the optimal parameters with last N samples of the validation dataset
    Search criteria: Find the setting combination with nr_cluster =1 and minimum nr_noise, min_radius and largest"""
    output = convert_to_dict_total()
    feature_dict = unique_symbol_dict_generation(output)
    if distance_metric == 2: # cross entropy
        laplace_smoothing = True

    else:
        laplace_smoothing = False

    tf_dict_adfa = convert_to_dict_tf(feature_dict, laplace_smoothing)
    validation_normal_dict = tf_dict_adfa["Validation_Data_Master"]

    validation_normal = []
    for k in validation_normal_dict.values():
        validation_normal.append(list(k.values()))
    validation_normal = np.array(validation_normal)
    nr_validation_normal = len(validation_normal)
    parameter_search_db = validation_normal[(nr_validation_normal - nr_parameter_search):]


    # eps_list =[i/100 for i in list(range(1, 100, 10))]
    # minp_list = range(180, 50, -10)
    eps_list = [i / 100 for i in list(range(1, 100, 20))]
    minp_list = range(180, 50, -20)

    for eps in eps_list:
        for minp in minp_list:
            labels_pred, n_clusters, n_noise = dbscan(parameter_search_db, eps, minp, distance_metric)


            print(n_clusters, n_noise, eps, minp)




def DL4LD_mitigation(app_name_list, portion_list, distance_metric, eqs, minP):

    op_acc_dict_total = {}
    op_acc_dict_s_total = {}

    dl4ld_dataset_split_para = {'couchdb': {'kernel': "rbf", 'ratio': 0.9},
                                'mongodb': {'kernel': "rbf", 'ratio': 0.9}}

    for app_name in app_name_list:
        kernel = dl4ld_dataset_split_para[app_name]['kernel']
        portion_train_test = dl4ld_dataset_split_para[app_name]['ratio']

        train_normal, train_attack, test_normal, test_attack = dl4ld_db_processing(app_name, portion_train_test)
        op_list = [0, 2, 3, 4]
        op_san_dict = {}
        for op in op_list:
            if op == 0:
                continue
            portion_tainted_dataset = LFP.label_flipping_db_generation(train_normal.copy(), train_attack.copy(), op,
                                                                       portion_list, kernel)
            portion_san_dict = {}

            for portion in portion_list:
                tainted_dataset = portion_tainted_dataset[portion][1]
                labels, n_clusters, n_noise = dbscan(tainted_dataset.copy(), eqs, minP, distance_metric)
                san_dataset = sanitization(tainted_dataset.copy(), labels)
                portion_san_dict[portion] = san_dataset
            op_san_dict[op] = portion_san_dict
        # op --> portion --> san_dict

        op_acc_dict = {}
        op_acc_dict_s = {}
        for op in op_list:
            if op == 0:
                acc = LFP.label_flipping_san(op_san_dict.copy(), train_normal.copy(), train_attack.copy(),
                                             test_normal.copy(), test_attack.copy(), op, portion_list, kernel)
                op_acc_dict[op] = acc
                op_acc_dict_s[op] = acc
            else:
                portion_dict_acc, portion_dict_acc_s = LFP.label_flipping_san(op_san_dict.copy(), train_normal.copy(),
                                                                              train_attack.copy(), test_normal.copy(),
                                                                              test_attack.copy(), op, portion_list, kernel)
                op_acc_dict[op] = portion_dict_acc
                op_acc_dict_s[op] = portion_dict_acc_s

        op_acc_dict_total[app_name] = op_acc_dict
        op_acc_dict_s_total[app_name] = op_acc_dict_s

        # plt.scatter_plot_err_s(op_acc_dict_total[app_name], op_acc_dict_s_total[app_name], portion_list)


def ADFA_LA_mitigation(nr_train_normal, eqs, minP, kernel,
                       portion_train_test, distance_metric, portion_list, dr_op, nr_dimension, attack_index_list):
    """

       distance_metric: describe how to calculate the distance between samples in the DBSCAN clutering algo:
       distance_metric = 1 ==> euclidean
       distance_metric =2 ==> cross entropy
        """

    op_acc_dict_total = {}
    op_acc_dict_s_total = {}
    for attack_index in attack_index_list:
        attack = attack_folder_list[attack_index]

        # square or euclidean
        if distance_metric != 2:
            """Step 1: Generate train_normal, train_attack, test_normal, test_attack into np arrays """
            laplace_smoothing = False
            train_normal, train_attack, test_normal, test_attack = adfa_db_preprocessing(attack, laplace_smoothing,
                                                                                         nr_train_normal, portion_train_test)
            """Step 2: Generate tainted dataset with specific portion and label flipping strategies """
            op_list = [0, 2, 3, 4]
            op_san_dict = {}
            for op in op_list:
                if op == 0:
                    continue
                portion_tainted_dataset = LFP.label_flipping_db_generation(train_normal, train_attack, op, portion_list,
                                                                           kernel)
                portion_san_dict = {}
                for portion in portion_list:
                    tainted_dataset = portion_tainted_dataset[portion][1]
                    labels, n_clusters, n_noise = dbscan(tainted_dataset, eqs, minP, distance_metric, dr_op, nr_dimension)
                    if n_clusters == 0:
                        raise ValueError("all the samples are detected as outliers")
                    san_dataset = sanitization(tainted_dataset, labels)
                    portion_san_dict[portion] = san_dataset
                op_san_dict[op] = portion_san_dict
            # op --> portion --> san_dict

        elif distance_metric == 2: # cross entropy
            train_normal, train_attack, test_normal, test_attack = adfa_db_preprocessing(attack, False,
                                                                                    nr_train_normal, portion_train_test)
            train_normal_l, train_attack_l, test_normal_l, test_attack_l = adfa_db_preprocessing(attack, True,
                                                                                         nr_train_normal,
                                                                                         portion_train_test)
            """Step 2: Generate tainted dataset with specific portion and label flipping strategies """
            # op_list = [0, 2, 3, 4]
            op_list = [0, 2, 3, 4]
            op_san_dict = {}
            for op in op_list:
                if op == 0:
                    continue
                portion_tainted_dataset = LFP.label_flipping_db_generation(train_normal, train_attack, op, portion_list,
                                                                           kernel)
                """tainited dataset with laplace smoothing, for dbscan filtering only"""
                portion_tainted_dataset_l = LFP.label_flipping_db_generation(train_normal_l, train_attack_l, op, portion_list,
                                                                           kernel)
                portion_san_dict = {}
                for portion in portion_list:
                    tainted_dataset = portion_tainted_dataset[portion][1]
                    """tainited dataset with laplace smoothing, for dbscan filtering only"""
                    tainted_dataset_l = portion_tainted_dataset_l[portion][1]
                    """Output the flags of each sample with TF + laplace smoothing"""

                    labels, n_clusters, n_noise = dbscan(tainted_dataset_l, eqs, minP, distance_metric, dr_op, nr_dimension)
                    if n_clusters == 0:
                        raise ValueError("all the samples are detected as outliers")

                    """Use the labels information and apply it to the original feature space (tf) """
                    san_dataset = sanitization(tainted_dataset, labels)

                    portion_san_dict[portion] = san_dataset
                op_san_dict[op] = portion_san_dict
            # op --> portion --> san_dict

        op_acc_dict = {}
        op_acc_dict_s = {}
        for op in op_list:
            if op == 0:
                acc = LFP.label_flipping_san(op_san_dict, train_normal, train_attack, test_normal, test_attack, op,
                                             portion_list, kernel)
                op_acc_dict[op] = acc
                op_acc_dict_s[op] = acc
            else:
                portion_dict_acc, portion_dict_acc_s = LFP.label_flipping_san(op_san_dict, train_normal,
                                                                              train_attack,
                                                                              test_normal, test_attack, op,
                                                                              portion_list,
                                                                              kernel)
                op_acc_dict[op] = portion_dict_acc
                op_acc_dict_s[op] = portion_dict_acc_s

        op_acc_dict_total[attack] = op_acc_dict
        op_acc_dict_s_total[attack] = op_acc_dict_s

    return op_acc_dict_total, op_acc_dict_s_total






def main():

    # For ADFA_LD dataset with eudlidean distance
    portion_list = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    # for fast test
    # portion_list = [0.05, 0.1]
    nr_train_normal = 300
    nr_parameter_search_db =300
    kernel = 'rbf'
    portion_train_test = 0.9
    improve_flag = True
    distance_metric = 1
    nr_dimension = 10
    attack_index_list = [0, 1, 2]

    # eqs = optimal_para_config[distance_metric][0]
    # minP = optimal_para_config[distance_metric][1]
    #
    # acc_dict_s_dr = {}
    # for dr_op in [0, 1, 2]:
    #     acc_dict, acc_dict_s = ADFA_LA_mitigation(nr_train_normal, eqs, minP, kernel, portion_train_test,
    #                                           distance_metric, portion_list, dr_op, nr_dimension, attack_index_list)
    #     acc_dict_s_dr[dr_op] = acc_dict_s
    #
    # plt.scatter_plot_err_dr(acc_dict, acc_dict_s_dr, portion_list, improve_flag)

    # with open('acc_dict_s_cross_entropy.json', 'w') as fp:
    #     json.dump(acc_dict_s_dm, fp)

    # with open('acc_dict_s_full.json', 'w') as fp_1:
    #     json.dump(acc_dict_s_dm, fp_1)


    # plt.scatter_plot_err(acc_dict, acc_dict_s_dm, portion_list, improve_flag)

    # parameter_search(nr_parameter_search_db, 2)


    #TODO: ADAPT THE PARAMETERS, REMOVE THE DEFAULT AUGUMENTS AND CHANGE THE configurations with app name
    eqs = 0.3
    minP = 100
    app_name_list = ['mongodb']
    DL4LD_mitigation(app_name_list, portion_list, distance_metric, eqs, minP)












if __name__ == "__main__":
    main()