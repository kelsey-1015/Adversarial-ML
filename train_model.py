"""This script is the main script, it parses the input tracefiles, trains the oc-svm model with input feature vectors
and compute the evaluation metrics"""

import trace_file_parser as tp
import oc_svm as oc
import json
import numpy as np
import argparse
from constants import *
import plot as pt

# TODO fix the test code stream

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


def result_filename(app_name, cross_label, oc_svm_kernel, feature_extraction, dr_flag):
    """return the .txt result output, normally achieve output from stdout"""
    if dr_flag:
        filename = cross_label+app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'_svd.txt'
    else:
        filename = cross_label+app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'txt'
    return filename


def result_labelname(oc_svm_kernel, feature_extraction, dr_flag):
    """Return labels for result output as json file"""
    if dr_flag:
        labelname = oc_svm_kernel+'_'+feature_extraction+'_svd'
    else:
        labelname = oc_svm_kernel+'_'+feature_extraction
    return labelname


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


def train_model(app_name, feature_dict_file, segment_length, filter_flag,
                oc_svm_kernel, feature_extraction, n_gram_length, attack_index):
    """Without fix nu and sengmentation length"""
    #TODO: move out the defination
    ad_attack_sq_list = [0, 1, 2, 3, 4]

    rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    if app_name == "ml0":
        rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack'][attack_index]
        print(rawtrace_file_attack)
    else:
        rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack']

    feature_extraction_index = FEATURE_VECTOR[feature_extraction]

    normal_set = dataset_concatenate(rawtrace_file_normal, feature_extraction_index, feature_dict_file,
                                               segment_length, filter_flag, n_gram_length)
    abnormal_set = extract_feature_vector(rawtrace_file_attack, feature_dict_file, feature_extraction_index,
                                             segment_length, filter_flag, n_gram_length)

    sq_dict = oc.attack_sq_loop(normal_set, abnormal_set, oc_svm_kernel, oc.nu, ad_attack_sq_list)
    # sq_dict = oc.parameter_search_test(normal_set, abnormal_set, oc_svm_kernel, oc.nu)


    return sq_dict


def train_model_fv_kernel(app_name, segment_length, filter_flag,
                          fv_list, kernel_list, n_gram_length, attack_index):
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear
    INPUT: dr_flag --> whether perform dimension reduction [truncted SVD]
           dr_dimension --> the number of perform dimension"""
    for fv in fv_list:
        for kernel in kernel_list:
            if fv == "N_GRAM":
                feature_dict_file = FEATURE_DICT_FILE[fv][app_name]
            else:
                feature_dict_file = FEATURE_DICT_FILE[fv]
            acc_dict = train_model(app_name, feature_dict_file, segment_length, filter_flag,
                                           kernel, fv, n_gram_length, attack_index)

    return acc_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--appname', type=str, default='couchdb', help='input the application name')
    # parser.add_argument('--dimension', type=int, default=5, help='input the dimension for trunctedSVD')
    args = parser.parse_args()
    app_name = args.appname
    # dr_dimension = args.dimension

    """Set different experimental settings """

    n_gram_length = 3
    segment_length = 30000
    # apply dimension reduction or not
    dr_flag_list = [False]
    # which feature extraction methods will be used['TF", "TFIDF", "N_GRAM"]
    fv_list = ["TF"]
    # which kernel to use ["rbf", "linear"]
    kernel_list = ["linear"]
    filter_flag = True
    dr_dimension = 15
    # indicate with attack is tested as abnormal data
    algorithm_dict_total = {}

    """For MongoDB, the attack_index only affects the ml0 app, so it can be set to 0 for other 2 applications"""
    attack_name = RAWTRACE_FILE[app_name]['attack']
    for attack_index in [1]:
        acc_dict = train_model_fv_kernel(app_name, segment_length, filter_flag, fv_list, kernel_list, n_gram_length,
                                         attack_index=attack_index)
        print(acc_dict)
        # base_line = acc_dict[0]
        # acc_1_mean = acc_dict[1][0]
        # acc_1_std = acc_dict[1][1]
        # acc_2 = acc_dict[2]
        # acc_3 = acc_dict[3]
        # acc_4 = acc_dict[4]
        # pt.scatter_plot_err(base_line, acc_1_mean, acc_1_std, acc_2, acc_3, acc_4)





if __name__ == "__main__":

    main()