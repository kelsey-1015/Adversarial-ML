"""This script includes the global variables"""

#
# # The filename of raw data for each application and attack
# RAWTRACE_FILE = {'couchdb': {'normal': ('raw_tracefile/couchdb_normal_11', 'raw_tracefile/couchdb_normal_12',
#                                         'raw_tracefile/couchdb_normal_13', 'raw_tracefile/couchdb_normal_14',
#                                                                            'raw_tracefile/couchdb_normal_15'),
#                              'attack': 'raw_tracefile/couchdb_attack'},
#                  'mongodb': {'normal': 'raw_tracefile/mongodb_normal', 'attack': 'raw_tracefile/mongodb_attack'},
#                  'ml0': {'normal': 'raw_tracefile/ml0_normal',
#                          'attack': ['raw_tracefile/ml0_attack_traces/ml0_attack', 'raw_tracefile/ml0_attack_traces/ml0_attack_1',
#                                     'raw_tracefile/ml0_attack_traces/attack_BIM', 'raw_tracefile/ml0_attack_traces/attack_CW',
#                                     'raw_tracefile/ml0_attack_traces/attack_FAB', 'raw_tracefile/ml0_attack_traces/attack_MIFGSM',
#                                     'raw_tracefile/ml0_attack_traces/attack_PGD', 'raw_tracefile/ml0_attack_traces/attack_PGDDLR',
#                                     'raw_tracefile/ml0_attack_traces/attack_Square',
#                                     'raw_tracefile/ml0_attack_traces/attack_TPGD']}}


# The filename of raw data for each application and attack, with less normal traces for couchdb
RAWTRACE_FILE = {'couchdb': {'normal': ('raw_tracefile/couchdb_normal_14'),
                             'attack': 'raw_tracefile/couchdb_attack'},
                 'mongodb': {'normal': 'raw_tracefile/mongodb_normal', 'attack': 'raw_tracefile/mongodb_attack'},
                 'ml0': {'normal': 'raw_tracefile/ml0_normal',
                         'attack': ['raw_tracefile/ml0_attack_traces/ml0_attack', 'raw_tracefile/ml0_attack_traces/ml0_attack_1',
                                    'raw_tracefile/ml0_attack_traces/attack_BIM', 'raw_tracefile/ml0_attack_traces/attack_CW',
                                    'raw_tracefile/ml0_attack_traces/attack_FAB', 'raw_tracefile/ml0_attack_traces/attack_MIFGSM',
                                    'raw_tracefile/ml0_attack_traces/attack_PGD', 'raw_tracefile/ml0_attack_traces/attack_PGDDLR',
                                    'raw_tracefile/ml0_attack_traces/attack_Square',
                                    'raw_tracefile/ml0_attack_traces/attack_TPGD']}}

# The list of unique syscall symbol with different applications and feature exaction methods
FEATURE_DICT_FILE = {'TF': "feature_vector_json/FEATURE_DICT.json", "TFIDF": "feature_vector_json/FEATURE_DICT.json",
                     "N_GRAM": {'couchdb': 'feature_vector_json/COUCHDB_FEATURE_DICT_NGRAM.json',
                                'mongodb': 'feature_vector_json/MONGODB_FEATURE_DICT_NGRAM.json',
                                 'ml0': 'feature_vector_json/ML0_FEATURE_DICT_NGRAM_FULL.json'}}


FEATURE_VECTOR = {'TF': 0, "TFIDF": 1, "N_GRAM": 2}
INFORMATION_STRING_1 = "# nu, FPR, TPR, std_FPR, std_TPR"
INFORMATION_STRING_2 = "# nu, FPR, TPR"

# The number of normal test data for mongodb application
SL_TN_number_mongodb = {1000: 746, 2000: 373, 5000: 149, 10000: 75, 15000: 50, 20000:37, 25000: 30, 30000: 25, 50000: 15}

"""The settings used to generate mongodb_benchmark"""
# segment_length_list = [20000, 50000]
# dr_flag_list = [True, False]
# fv_list = ['TF', 'TFIDF', 'N_GRAM']
# kernel_list = ["linear"]
# filter_flag = False

"""The settings for normal test"""
# segment_length_list = [50000]
# dr_flag_list = [True, False]
# fv_list = ['TF']
# kernel_list = ["linear"]
# filter_flag = False

"""The full settings"""
# segment_length_list = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]
# dr_flag_list = [True, False]
# fv_list = ['TF', 'TFIDF', 'N_GRAM']
# kernel_list = ["linear", "rbf"]
# filter_flag = False

# execution time for mongodb with SL = 30000
exe_time_ngram_linear = [490.3, 330.1, 15.8]
exe_time_ngram_rbf = [417.7, 327.0, 15.7]
exe_time_tf_linear = [17.4, 74.9, 8.2]
exe_time_tf_rbf = [16.9, 71.3, 8.3]


# FEATURE_DICT_FILE = {'TF': "feature_vector_json/FEATURE_DICT.json", "TFIDF": "feature_vector_json/FEATURE_DICT.json",
#                      "N_GRAM": {'couchdb': 'feature_vector_json/COUCHDB_FEATURE_DICT_NGRAM.json',
#                                 'mongodb': 'feature_vector_json/MONGODB_FEATURE_DICT_NGRAM.json',
#                                  'ml0': 'feature_vector_json/ML0_FEATURE_DICT_NGRAM.json'}}

