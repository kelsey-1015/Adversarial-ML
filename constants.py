"""This script includes the global variables"""

portion_list = [0.05, 0.075, 0.1, 0.125,  0.15, 0.175, 0.2]


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





