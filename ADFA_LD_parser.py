import os


"""This file parsing the ADFA_LD dataset and """

path = "/Users/lu/PycharmProjects/Adversarial ML/ADFA_LD/"
dataset_list = ["Attack_Data_Master", "Training_Data_Master", "Validation_Data_Master"]
dataset_numfile_dict= {"Training_Data_Master": 833, "Validation_Data_Master": 4372}
attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]


def text_filename_contruction(dataset_name, numb):
    """For training and validating dataset"""
    if dataset_name == dataset_list[1]: # training_data_normal
        text_filename = path + "Training_Data_Master/UTD-" + numb + ".txt"
    elif dataset_name == dataset_list[2]:
        text_filename = path + "Validation_Data_Master/UVD-" + numb + ".txt"
    return text_filename


def convert_to_dict(dataset_name, num_file):
    """training, validation folders"""
    numb_list_dict = {}
    for k in range(1, num_file+1):
        numb = format(k, '04')
        text_filename = text_filename_contruction(dataset_name, numb)
        text_file = open(text_filename)
        trace = text_file.read()
        word_list = [int(i) for i in trace.split()]
        numb_list_dict[numb] = word_list
    return numb_list_dict


def list_fre_dict(list, feature_dict, laplace_smoothing, normalization_flag=True):
    """map a list into a count dict"""
    for i in list:
        feature_dict[i] += 1
    # print("P: ", feature_dict)
    if normalization_flag:
        if laplace_smoothing:
            total = sum(feature_dict.values()) + len(feature_dict)
            factor = 1 / total
            feature_dict = {k: (v + 1) * factor for k, v in feature_dict.items()}
        else:

            sum_value = sum(feature_dict.values())
            for k, v in feature_dict.items():
                feature_dict[k] = float(v/sum_value)

    return feature_dict


def convert_to_dict_tf(feature_dict, laplace_smoothing):
    """parse adfa_ld dataset and construct a dictionary of list"""
    file_tf_dict = {}
    for dataset in dataset_list:
        file_tf_dict[dataset] = {}

        if dataset == "Attack_Data_Master":
            for attack_name in attack_folder_list:
                file_tf_dict[dataset][attack_name]={}
                subfold_path = path+dataset+"/"+attack_name
                for text_filename in os.listdir(subfold_path):
                    text_filename_full = subfold_path + "/" + text_filename
                    text_file = open(text_filename_full)
                    trace = text_file.read()
                    word_list = [int(i) for i in trace.split()]
                    # print(feature_dict)
                    word_fr_dict = list_fre_dict(word_list, feature_dict.copy(), laplace_smoothing)
                    file_tf_dict[dataset][attack_name][text_filename] = word_fr_dict
        else:
            file_tf_dict[dataset] = {}
            subfold_path = path + dataset
            for text_filename in os.listdir(subfold_path):
                text_filename_full = subfold_path + "/" + text_filename
                text_file = open(text_filename_full)
                trace = text_file.read()
                word_list = [int(i) for i in trace.split()]
                # print(feature_dict)
                word_fr_dict = list_fre_dict(word_list, feature_dict.copy(), laplace_smoothing)
                file_tf_dict[dataset][text_filename] = word_fr_dict
    return file_tf_dict


def convert_to_dict_total():
    """parse adfa_ld dataset and construct a dictionary of list"""
    file_symbol_dict = {}
    for dataset in dataset_list:
        file_symbol_dict[dataset] = {}

        if dataset == "Attack_Data_Master":
            for attack_name in attack_folder_list:
                file_symbol_dict[dataset][attack_name]={}
                subfold_path = path+dataset+"/"+attack_name
                for text_filename in os.listdir(subfold_path):
                    text_filename_full = subfold_path + "/" + text_filename
                    text_file = open(text_filename_full)
                    trace = text_file.read()
                    word_list = [int(i) for i in trace.split()]
                    file_symbol_dict[dataset][attack_name][text_filename] = word_list
        else:
            file_symbol_dict[dataset] = {}
            subfold_path = path + dataset
            for text_filename in os.listdir(subfold_path):
                text_filename_full = subfold_path + "/" + text_filename
                text_file = open(text_filename_full)
                trace = text_file.read()
                word_list = [int(i) for i in trace.split()]
                file_symbol_dict[dataset][text_filename] = word_list

    return file_symbol_dict



def unique_symbol_dict_generation(file_symbol_dict):
    unique_symbol_list = []
    unique_symbol_list = extract_nested_values(file_symbol_dict, unique_symbol_list)
    feature_dict = dict(zip(unique_symbol_list, [0]*len(unique_symbol_list)))
    return feature_dict



def extract_nested_values(file_symbol_dict, unique_symbol_list):
    """extract all unique values in a nested dict"""
    for k, v in file_symbol_dict.items():
        if isinstance(v, dict):
            unique_symbol_list = extract_nested_values(v, unique_symbol_list)
        else:
            # print(v)
            # unique_symbol_list = list(unique_symbol_list)
            unique_symbol_list = unique_symbol_list + v
            unique_symbol_list = set(unique_symbol_list)
            unique_symbol_list = list(unique_symbol_list)

    return unique_symbol_list



def train_model(train_normal, train_attack, test_normal, test_attack, oc_svm_kernel, ad_attack_sq_list, portion_list):
    """acc_dict = {0(base_line): float ; 1 (Random Noise): {0 (mean): list; 1(std):list};
    2 (FF):portion_acc_dict ; 3 (NF): portion_acc_dict; 4 (ALFA):portion_acc_dict }"""

    acc_dict = lfa.attack_sq_loop(train_normal, train_attack, test_normal, test_attack, oc_svm_kernel, lfa.nu,
                                  ad_attack_sq_list, portion_list)
    return acc_dict




def main():

    kernel = "rbf"
    ad_attack_sq_list = [0, 1, 2, 3, 4]

    title_text = "Performance degradation with ADFA_LD dataset (TF, Gaussion Kernel, adversarial samples from different " \
                 "attacks))"
    output = convert_to_dict_total()
    feature_dict = unique_symbol_dict_generation(output)
    tf_dict = convert_to_dict_tf(feature_dict)
    attack_acc_dict = {}

    # # train_normal, train_attack, test_normal, test_attack = lfa.data_preprocessing_multiple_attack(tf_dict,
    #                                                                                               attack_folder_list[0],
    #                                                                               attack_folder_list[1:])

    for attack_name in attack_folder_list:
        attack_test = attack_name
        attack_train_list = attack_folder_list.copy()
        attack_train_list.remove(attack_test)

        train_normal, train_attack, test_normal, test_attack = lfa.data_preprocessing_multiple_attack(tf_dict,
                                                                                                      attack_test,
                                                                                                      attack_folder_list)
        acc_dict = train_model(train_normal, train_attack, test_normal, test_attack, kernel, ad_attack_sq_list,
                               portion_list)

        attack_acc_dict[attack_name] = acc_dict

    print(attack_acc_dict)

    # pt.scatter_plot_err(attack_acc_dict, portion_list, title_text)



if __name__ == "__main__":

    main()