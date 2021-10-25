import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json



def plot_algorithm(data_dict, color_list, linestyle_list, segment_length_list):
    plt.figure()
    index = 0
    for key, list in data_dict.items():
        color_s = color_list[index]
        ls_s = linestyle_list[index]
        plt.plot(segment_length_list, list[0], marker='o', label="TPR_"+key, color=color_s, linestyle=ls_s,
                 markersize=10, linewidth=4)
        plt.plot(segment_length_list, list[1], marker='x', label="FPR_"+key, color=color_s, linestyle=ls_s,
                 markersize=10, linewidth=4)
        index += 1
    plt.legend(prop={'size': 10}, ncol=2, handleheight=2.4, labelspacing=0.1)
    # plt.legend(prop={'size': 10}, loc=0)
    plt.xlabel("Segment Length (# system call)", fontsize=20)
    plt.ylabel("TPR and FPR", fontsize=20)
    plt.grid()
    # plt.title("TPR and FPR with different segment window size for COUCHDB", fontsize=16)
    plt.show()


def data_process_auc(json_file, nu='0.01'):
    dict = read_data(json_file)
    dict_reform = {}
    for k, v in dict.items():
        print("label name: ", k)
        segment_list = []
        auc_list = []
        for k1, v1 in v.items():
            segment_list.append(int(k1))
            auc = v1[nu]
            auc_list.append(auc)
        dict_reform[k] = auc_list
    return dict_reform, segment_list


def data_process_roc(json_file_list, k_fold_index,  nu='0.01', segment_length='30000', label_name="rbf_N_GRAM"):
    """pre-processing for roc json file for couchdb/mongodb/ml0"""
    color_list = ['b', 'g', 'r']
    line_style_list = ['-', '-.', ':']

    key_num = len(json_file_list)
    key_index = 0

    for json_file in json_file_list:
        dict = read_data(json_file)
        v = dict[label_name]
        fpr_list = v[segment_length][nu][0]
        tpr_list = v[segment_length][nu][1]
        roc_curve(fpr_list, tpr_list, key_index, key_num, k_fold_index, color_list, line_style_list)
        # roc_curve(fpr_list, tpr_list, key_index, key_num, k_fold_index)
        key_index += 1

    return fpr_list, tpr_list


def data_process_roc_v0(json_file, k_fold_index,  nu='0.01', segment_length='30000', label_name="rbf_N_GRAM"):
    """pre-processing for roc json file with multiple attacks"""
    # color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    # line_style_list = ['-', '--', '-', '-', '-.', ':']
    dict = read_data(json_file)
    key_num = len(dict.items())
    key_index = 0
    for k, v in dict.items():
        v = v[label_name]
        fpr_list = v[segment_length][nu][0]
        tpr_list = v[segment_length][nu][1]
        # roc_curve(fpr_list, tpr_list, key_index, key_num, color_list, line_style_list)
        roc_curve(fpr_list, tpr_list, key_index, key_num, k_fold_index)
        key_index += 1
    return fpr_list, tpr_list


def roc_curve(FPR_list, TPR_list, key_index, key_num, k_fold_index, color_list, linestyle_list):
    # plt.figure()
    # for index in range(len(FPR_list)):
    #     plt.plot(FPR_list[index], TPR_list[index], linestyle=':', linewidth=4)
    """mongodb,index=6"""
    color_s = color_list[key_index]
    ls_s = linestyle_list[key_index]
    plt.scatter(FPR_list[k_fold_index], TPR_list[k_fold_index], color=color_s)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if key_index == key_num-1:
        # plt.legend(["linear_TF", "rbf_TF", "linear_TFIDF", "rbf_TFIDF", "linear_N_GRAM", "rbf_N_GRAM"])
        plt.grid()
        plt.show()



def plot_algorithm_auc(data_dict, color_list, linestyle_list, segment_length_list):
    plt.figure()
    index = 0
    for key, list in data_dict.items():
        color_s = color_list[index]
        ls_s = linestyle_list[index]
        plt.plot(segment_length_list, list, label=key, color=color_s, linestyle=ls_s,
                 markersize=10, linewidth=4)
        index += 1
    plt.legend(prop={'size': 10}, ncol=2, handleheight=2.4, labelspacing=0.1)
    # plt.legend(prop={'size': 10}, loc=0)
    plt.xlabel("Segment Length (# system call)", fontsize=20)
    plt.ylabel("AUC value", fontsize=20)
    plt.grid()
    plt.show()


# def plot_fpr_reduction(fpr_original_list_1, fpr_new_list_1,
#                        fpr_original_list_2, fpr_new_list_2,segment_length_list):
#
#     fig, axs = plt.subplots(2)
#     # fig.suptitle('Vertically stacked subplots')
#
#     axs[0].plot(segment_length_list, fpr_original_list_1, marker='x', color='red', label="Original FPR for TF",  linewidth=2)
#     axs[0].plot(segment_length_list, fpr_new_list_1, marker='x', color='blue', label="Reduced FPR for TF", linewidth=2)
#     axs[1].plot(segment_length_list, fpr_original_list_2, marker='x', color='red', label="Original FPR for NGRAM", linewidth=2)
#     axs[1].plot(segment_length_list, fpr_new_list_2, marker='x', color='blue', label="Reduced FPR for NGRAM", linewidth=2)
#
#     axs[0].legend(prop={'size': 8}, loc='best')
#     axs[1].legend(prop={'size': 8}, loc='best')
#     axs[0].set_xlabel("Segment Length (# system call)")
#     axs[1].set_xlabel("Segment Length (# system call)")
#     axs[0].set_ylabel("False Positive Rate (FPR)")
#     axs[1].set_ylabel("False Positive Rate (FPR)")
#     axs[0].set_ylim([-0.005, 0.035])
#     axs[1].set_ylim([-0.005, 0.035])
#     axs[0].grid()
#     axs[1].grid()
#
#
#     # plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
#     plt.show()

# def plot_fpr_reduction(tpr_original_list, tpr_new_list, fpr_original_list, fpr_new_list, segment_length_list):
#
#
#
#     plt.figure()
#     # fig.suptitle('Vertically stacked subplots')
#
#     plt.plot(segment_length_list, tpr_original_list, marker='o', color='red', label="Original TPR",  linewidth=2)
#     plt.plot(segment_length_list, tpr_new_list, marker='o', color='blue', label="TPR after Filtering", linewidth=2)
#     plt.plot(segment_length_list, fpr_original_list, marker='x', color='red', label="Original FPR", linewidth=2)
#     plt.plot(segment_length_list, fpr_new_list, marker='x', color='blue', label="FPR after Filtering", linewidth=2)
#
#     plt.legend(prop={'size': 10}, handleheight=2.4, labelspacing=0.1)
#     # plt.legend(prop={'size': 10}, loc=0)
#     plt.xlabel("Segment Length (# system call)", fontsize=20)
#     plt.ylabel("TPR and FPR value", fontsize=20)
#
#     plt.grid()
#     # plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
#     plt.show()


def plot_fpr_reduction(tpr_original_list, tpr_new_list, fpr_original_list, fpr_new_list, segment_length_list):

    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')

    axs[0].plot(segment_length_list, tpr_original_list, marker='o', color='red', label="Original TPR",  linewidth=4)
    axs[0].plot(segment_length_list, tpr_new_list, marker='o', color='blue', linestyle=':',
                label="TPR after Filtering", linewidth=4)
    axs[1].plot(segment_length_list, fpr_original_list, marker='x', color='red', label="Original FPR", linewidth=4)
    axs[1].plot(segment_length_list, fpr_new_list, marker='x', color='blue', linestyle=':', label="FPR after Filtering", linewidth=4)

    axs[0].legend(prop={'size': 12}, loc='best')
    axs[1].legend(prop={'size': 12}, loc='best')
    axs[0].set_xlabel("Segment Length (# system call)", fontsize=18)
    axs[1].set_xlabel("Segment Length (# system call)", fontsize=18)
    axs[0].set_ylabel("TPR", fontsize=18)
    axs[1].set_ylabel("FPR", fontsize=18)
    axs[0].set_ylim([0.975, 1.005])
    axs[1].set_ylim([-0.005, 0.035])
    axs[0].grid()
    axs[1].grid()

    plt.show()


    # plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
#     plt.show()

def read_data(json_filename):
    with open(json_filename, "r") as read_file:
        dict = json.load(read_file)
    return dict


def data_process(dict, nu):
    """Pre-process the json file so that it can feed into the plot function"""
    dict_reform_total = {}
    for k, v in dict.items():
        tpr_list = []
        fpr_list = []
        segment_list = []
        for k1, v1 in v.items():
            # print(k, k1)
            fpr = v1[nu][0]
            tpr = v1[nu][1]
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            segment_list.append(int(k1))
        dict_reform_total[k] = [tpr_list, fpr_list]
    # split the data to svm and svm_pca
    dict_reform_svd = {}
    dict_reform = {}
    for k, v in dict_reform_total.items():
        ks = k.split("_")
        # print(ks)
        if ks[-1] == 'svd':
            dict_reform_svd[k] = v
        else:
            dict_reform[k] = v
    return dict_reform, dict_reform_svd,  segment_list




def scatter_plot_err(baseline_value, acc_dict_mean_1, acc_dict_error_1, acc_2, acc_3, acc_4):
    """This function is used to plot the performance degradation due to advery"""
    lw = 3
    # portion_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    portion_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
    fig.suptitle("COUCHDB: SETTING: TF, Linear, 30000")

    baseline = [baseline_value] * len(portion_list)
    ACC_1 = list(acc_dict_mean_1.values())
    error_1 = list(acc_dict_error_1.values())
    ax[0].plot(portion_list, baseline, linewidth=lw)
    ax[0].errorbar(portion_list, ACC_1, yerr=error_1, linewidth=lw)
    ax[0].set_title('Uniform Random Flipping')
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Portion")
    ax[0].yaxis.grid(True)

    ACC_2 = list(acc_2.values())
    ax[1].plot(portion_list, baseline, linewidth=lw)
    ax[1].errorbar(portion_list, ACC_2, linewidth=lw)
    ax[1].set_title('Furthest-first Flipping')
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Portion")
    ax[1].yaxis.grid(True)

    ACC_3 = list(acc_3.values())
    ax[2].plot(portion_list, baseline, linewidth=lw)
    ax[2].errorbar(portion_list, ACC_3, linewidth=lw)
    ax[2].set_title('Nearest-first Flipping')
    ax[2].set_ylabel("Accuracy")
    ax[2].set_xlabel("Portion")
    ax[2].yaxis.grid(True)

    ACC_4 = list(acc_4.values())
    ax[3].plot(portion_list, baseline, linewidth=lw)
    ax[3].errorbar(portion_list, ACC_4, linewidth=lw)
    ax[3].set_title('ALFA')
    ax[3].set_ylabel("Accuracy")
    ax[3].set_xlabel("Portion")
    ax[3].yaxis.grid(True)


    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()



def bar_plot_err(baseline_value, acc_dict_mean_1, acc_dict_error_1, acc_2, acc_3):
    """This function is used to plot the performance degradation due to advery"""
    portion_list = ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]

    x_pos = np.arange(len(portion_list))
    x_pos_benchmark = np.arange(-1, 6)

    y_bench = [baseline_value] * len(x_pos_benchmark)
    # Build the plot
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)


    ACC_1 = list(acc_dict_mean_1.values())
    error_1 = list(acc_dict_error_1.values())
    ax[0].plot(x_pos_benchmark, y_bench)
    ax[0].bar(x_pos, ACC_1, yerr=error_1, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Portion")
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(portion_list)
    # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
    ax[0].yaxis.grid(True)

    ACC_2 = list(acc_2.values())
    ax[1].plot(x_pos_benchmark, y_bench)
    ax[1].bar(x_pos, ACC_2, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Portion")
    ax[1].set_xticks(x_pos)
    ax[1].set_xticklabels(portion_list)
    ax[1].set_title('Furthest First Flip')
    ax[1].yaxis.grid(True)


    ACC_3 = list(acc_3.values())
    ax[2].plot(x_pos_benchmark, y_bench)
    ax[2].bar(x_pos, ACC_3, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax.set_ylabel("Accuracy")
    # ax.set_xlabel("Portion")
    ax[2].set_xticks(x_pos)
    ax[2].set_xticklabels(portion_list)
    ax[2].yaxis.grid(True)

    plt.ylim(0, 1.05)
    plt.show()



def bar_plot():
    labels = ['MongoDB', 'CouchDB', "Image Classification"]

    exe_time_ngram_linear = [490.3, 330.1, 15.8]
    exe_time_ngram_rbf = [417.7, 327.0, 15.7]
    exe_time_tf_linear = [17.4, 74.9, 8.2]
    exe_time_tf_rbf = [16.9, 71.3, 8.3]
    exetime_list_new = execution_time_process([exe_time_ngram_linear, exe_time_ngram_rbf, exe_time_tf_linear,
                                               exe_time_tf_rbf])
    exe_time_ngram_linear = exetime_list_new[0]
    exe_time_ngram_rbf = exetime_list_new[1]
    exe_time_tf_linear = exetime_list_new[2]
    exe_time_tf_rbf = exetime_list_new[3]


    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, exe_time_ngram_linear, width, color ='blue', label='N_GRAM_linear')
    rects2 = ax.bar(x - width/2, exe_time_tf_linear, width, color='red', label='TF_linear')
    rects3 = ax.bar(x + width/2, exe_time_ngram_rbf, width, color='blue', label='N_GRAM_rbf', hatch="//")
    rects4 = ax.bar(x + width*3/2, exe_time_tf_rbf, width, color='red', label='TF_rbf', hatch="//")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution Time ($T_0$)', fontsize=20)
    # ax.set_title('Execution Time for Mongodb with segment length = 30000')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.grid()
    ax.legend(fontsize=16)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()

    plt.show()

def main():


    """BLOCK 1:  is used to plot the FPR/TPR values of 3 APPs (Fig 3(a), 3(b), 3(c) in the paper)"""
    # segment_length_list = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]

    # json_filename = "json_result/couchdb_mix.json"
    # output_dict = read_data(json_filename)
    # color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    # line_style_list = ['-', '--', '-', '-', '-.', ':']
    # #
    # for nu in [0.01]:
    #     nu = str(nu)
    #     dict_reform, dict_reform_svd, segment_length_list = data_process(output_dict, nu)
    #     plot_algorithm(dict_reform, color_list, line_style_list, segment_length_list)

    """End of BLOCK 1"""

    """BLOCK 2:  used to plot the execution time of 3 APPs (Fig 3(d) in the paper), the plotting data is embedded in the
    code and also available in RESULTS_DATA_FOR_PLOTTING/EXECUTION_TIME_RESULTS.txt"""
    # bar_plot()
    """End of BLOCK 2"""



    # app_name = 'ml0'
    # rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    # feature_dict_file = FEATURE_DICT_FILE["TF"]
    # feature_vector_list = tm.extract_feature_vector(rawtrace_file_normal, feature_dict_file, 1, 2000, False)
    # PCA_dimension(feature_vector_list)


    # json_filename = 'ml0_performance.json'
    # dict_reform, seg_list = data_process_auc(json_filename)
    # color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    # line_style_list = ['-', '--', '-', '-', '-.', ':']
    # plot_algorithm_auc(dict_reform, color_list, line_style_list, seg_list)

    # json_filename = 'ml0_multiple_attack_roc.json'
    # json_filename_1 = 'roc_result/couchdb_performance.json'
    # json_filename_2 = 'roc_result/mongodb_performance.json'
    # json_filename_3 = 'roc_result/ml0_performance.json'
    # json_filename_list = [json_filename_1, json_filename_2, json_filename_3]
    #
    #
    # for kfold_index in range(10):
    #     fpr_list, tpr_list = data_process_roc(json_filename_list, kfold_index)
    # roc_curve(fpr_list, tpr_list)


if __name__ == "__main__":

    main()