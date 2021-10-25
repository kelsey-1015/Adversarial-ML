import os
import csv
import math
import json
import ast
from constants import *
from sklearn.decomposition import PCA

"""This script parse the raw traces and extract the keys of the [item- pair] dictionary for different feature 
vectors"""

NGRAM_LENGTH = 6


# The following syscalls may happen in the raw tracefile, but they are not valid. Not in the Feature vector
# the system calls exist in the raw trace file but not in the feature dict
INVALID_SYSCALL_LIST = ['procexit', 'signaldeliver', 'prlimit', '<unknown>', "container"]

# the following syscalls exist in feature vectors, but may be filtered out for performance reasons, they are valid syscalls
FILTER_SYSCALL_LIST = ["futex", "sched_yield" ]


# Python code to merge dict using update() method
def merge(dict1, dict2):
    dict2.update(dict1)
    return dict2


def normalization(input_list):
    """Change number of occurrence input frequency distribution for a nested list, remove the effect of the last element
    , which is the segment sequence"""
    output_list = []
    for fv in input_list:
        segment_sequence = fv[-1]
        # remove the last element for normalization
        fv = fv[:-1]
        # change str to int
        fv = [int(c) for c in fv]
        fv_norm = [float(c)/sum(fv) for c in fv]
        fv_norm.append(segment_sequence)
        output_list.append(fv_norm)
    return output_list


def strlist_2_list(strlist):
    """ reverse str(list) to a list; e.g '[1, 2, 4]' --> [1, 2, 4]"""
    x = ast.literal_eval(strlist)
    x = [n.strip() for n in x]
    return x


def common_element(a, b):
    """ compute if two lists have common memebers"""
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


def generate_n_gram(n_gram, key_action, n_gram_length):
    """ This function generates n-grams of specific length"""

    if len(n_gram) < n_gram_length:
        n_gram.append(key_action)
        return n_gram

    elif len(n_gram) == n_gram_length:
        n_gram = n_gram[1:]
        n_gram.append(key_action)
        return n_gram

    else:
        raise ValueError


def n_gram_dict(filename, distinct_ngram_list, n_gram_length):
    """ This function extracts or update occurring distinct syscalls in the raw tracefile
    INPUT: filename --> new input"""
    n_gram = []
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError

                if sys_call in INVALID_SYSCALL_LIST: # skip the line IF the corresponding syscall is invalid
                    continue

                n_gram = generate_n_gram(n_gram, sys_call, n_gram_length)
                if len(n_gram) == n_gram_length:
                    n_gram_s = str(n_gram)
                    if n_gram_s not in distinct_ngram_list:
                        distinct_ngram_list.append(n_gram_s)

    keys = distinct_ngram_list
    values = [0] * len(keys)
    distinct_ngram_dict = dict(zip(keys, values))

    return distinct_ngram_list, distinct_ngram_dict


def fixed_length_feature_dict(filename, distinct_syscall_list=[]):
    """ This function extracts or update occurring distinct syscalls in the raw tracefile
    INPUT: filename --> new input
    distinct_syscall_list --> existing system call list"""
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError
                if sys_call not in distinct_syscall_list:
                    distinct_syscall_list.append(sys_call)
    keys = distinct_syscall_list
    values = [0] * len(keys)
    feature_dict = dict(zip(keys, values))
    if 'container' in feature_dict.keys():
        del feature_dict['container']
    return feature_dict


def parse_trace_ngram(filename, feature_dict, num_separate, filter_flag, n_gram_length):
    """This function changes traces into fix length n-gram feature vectors
    INPUT: filename --> raw tracefiles
    feature_dict: The fixed length feature vector
    OUTPUT:
    feature_vector_list: a nested list of feature vectors, the sequence number is appended in the last element of each
    feature vector; """
    feature_vector_list = []
    segment_sequence = 0
    fv_index = 0
    # skip this trace if it is empty
    if os.stat(filename).st_size == 0:
        raise ValueError("The input file %s is empty!" % filename)
        return -1
    n_gram = []
    syscall_index = 0
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                syscall_index += 1
                # print(syscall_index)
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError

                if filter_flag:
                    if sys_call not in FILTER_SYSCALL_LIST: # if the action not belongs to one of the followings
                        n_gram = generate_n_gram(n_gram, sys_call, n_gram_length)
                        if len(n_gram) == n_gram_length:
                            n_gram_s = str(n_gram)
                            try:
                                feature_dict[n_gram_s] += 1
                            except KeyError:  # raise key error if the ngram does not exist
                                pass
                else:
                    n_gram = generate_n_gram(n_gram, sys_call, n_gram_length)

                    if len(n_gram) == n_gram_length:
                        n_gram_s = str(n_gram)
                        try:
                            feature_dict[n_gram_s] += 1
                        except KeyError: # raise key error if the ngram does not exist
                            pass

                if syscall_index == num_separate:
                    # print('seprate the tracefile')
                    segment_sequence += 1
                    syscall_index = 0
                    feature_vector = list(feature_dict.values())
                    feature_vector.append(segment_sequence)

                    #  skip if all the elements all zero
                    all_zero_flag = all([o == 0 for o in feature_vector[:-1]])
                    if not all_zero_flag:
                        fv_index += 1
                        feature_vector_list.append(feature_vector)

                    # reset dict values to zeros
                    feature_dict = dict.fromkeys(feature_dict, 0)
    return feature_vector_list


def parse_trace_tmp(filename, feature_dict, num_separate, filter_flag):
    """This function changes traces into feature vectors. The raw trace is segmented with length window or time window
    OUTPUT: occurrence_dict: The no. of occurrences of features [syscall, ngram] in all traces of the database
            feature_vector_list: a nested list of feature vectors, the sequence number is appended in the last element
            of each feature vector; feature vector: occurrence number of system calls in a segment"""

    feature_vector_list = []
    # compute the occurrence of syscalls in the total database, currently only on the normal data; used for tf-idf
    occurrence_list = [0]*len(feature_dict.values())
    # total number of traces, for tf-idf
    segment_sequence = 0
    # skip this trace if it is empty
    if os.stat(filename).st_size == 0:
        raise ValueError("The input file %s is empty!" % filename)
        return -1

    syscall_index = 0
    with open(filename) as f:
        for line in f:
            if line.strip():  # if the line is not empty
                syscall_index += 1
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                    sys_call_time = line_list[1]
                except:
                    raise ValueError

                if filter_flag:
                    # if the action not belongs to one of the followings
                    if sys_call not in FILTER_SYSCALL_LIST:
                        try:
                            feature_dict[sys_call] += 1
                        except:
                            if sys_call not in INVALID_SYSCALL_LIST:
                                print("Unexpected system call: ", sys_call)
                            pass
                else:
                    try:
                        feature_dict[sys_call] += 1
                    except:
                        pass

                if syscall_index == num_separate:
                    # print('seprate the tracefile')
                    segment_sequence += 1
                    syscall_index = 0
                    feature_vector = list(feature_dict.values())

                    # validate if a specific syscall/ngram occurs in the feature vector, used to calculate idf (inverse
                    # document frequency)_
                    feature_vector_flag_tmp = [1 if c > 0 else 0 for c in feature_vector]
                    # element-wise addition of symbol occurrence number in the current trace and the historic data
                    occurrence_list = [sum(x) for x in zip(feature_vector_flag_tmp, occurrence_list)]
                    occurrence_dict = dict.fromkeys(feature_dict, 0)
                    # update the dict
                    occurrence_dict.update(zip(occurrence_dict, occurrence_list))

                    # attach the segment sequence into the last element of the feature vector
                    feature_vector.append(segment_sequence)
                    # compute the feature_vector list for each tracefile (separated with lenghth N), only remain those
                    # dictinct feature vectors
                    #  skip if all the elements all zero
                    all_zero_flag = all([o==0 for o in feature_vector[:-1]])
                    if not all_zero_flag:
                        # print("single_feature_vector: ", feature_vector)
                        feature_vector_list.append(feature_vector)

                    feature_dict = dict.fromkeys(feature_dict, 0)

    return feature_vector_list, occurrence_dict, segment_sequence


def df_idf(feature_vector_list, occurrence_dict, N):
    """ This function generates nested feature vectors weighted tf-idf"""
    p_list = []
    fv_idf_list = []
    for v in occurrence_dict.values():
        v = float(v)
        p = math.log(N/(v+1), 10)
        p_list.append(p)

    for fv in feature_vector_list:
        segment_sequence = fv[-1]
        fv = fv[:-1]
        fv_idf = [a*b for a, b in zip(p_list, fv)]
        fv_idf.append(segment_sequence)
        fv_idf_list.append(fv_idf)
    return fv_idf_list


def feature_vector_csv_generator(feature_vector_list, csv_filename):
    """Write the feature vector list into csv file"""

    with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(feature_vector_list)


def generate_ngram_dict_json(rawtracefile_list, json_filename, n_gram_length):
    """Generate the json file for n_gram feature vectors as a json file
     INPUT --> a list of raw tracefiles name
     OUTPUT --> a json file"""
    distinct_ngram_list = []
    for filename in rawtracefile_list:
        ngram_list, ngram_dict = n_gram_dict(filename, distinct_ngram_list, n_gram_length)
        distinct_ngram_list = ngram_list
    print(len(list(ngram_dict.keys())))
    json.dump(ngram_dict, open(json_filename, 'w'))


def main():
    app_name = 'ml0'
    rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack']
    # print(rawtrace_file_normal)
    # print(rawtrace_file_attack)

    rawtrace_file = rawtrace_file_attack.copy()

    rawtrace_file.append(rawtrace_file_normal)

    # print("here", rawtrace_file)

    n_gram_length = 3

    json_filename = "ML0_FEATURE_DICT_NGRAM_FULL.json"
    #
    generate_ngram_dict_json(rawtrace_file, json_filename, n_gram_length)







if __name__ == "__main__":
    main()
