import numpy as np
import matplotlib.pyplot as plt
import json

def scatter_plot_err_json(acc_dict_total, acc_dict_s_dm,  portion_list, improve_flag):
    """This function is used to plot the performance degradation due to advery
    improve_flag: whether this plot contains results after sanitation
    improve_flag == 1 ==> contains"""
    lw = 3
    nr_row = len(acc_dict_total)
    # print(acc_dict_total)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser", "Java Meterpreter", "Web Shell"]
    attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
    sq_dist_metric_dict = {1: "euclidean", 2: "cross entropy", 3: "square distance"}
    # app_name_list = ["CouchDB", "MongoDB"]


    # loop over datset attack
    attack_index = 0
    for attack_dataset in attack_folder_list:
        acc_dict = acc_dict_total[attack_dataset]
        baseline_value = acc_dict['0']
        baseline_value = int(baseline_value)
        baseline = [baseline_value] * len(portion_list)
        # acc_dict_s = acc_dict_s_total[attack_dataset]

        for j in [1, 2, 3]: # loop over poisoning strategy
            # For each subplot

            acc_d = acc_dict[str(j+1)]
            acc = list(acc_d.values())

            # acc_d_s = acc_dict_s[j+1]
            # acc_s = list(acc_d_s.values())

            ax[attack_index, j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Base line')
            ax[attack_index, j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
            if improve_flag:
                # print(acc_dict_s_dm.keys())
                print("Poisoning Strategy: ", j)
                for k, acc_dict_s_total in acc_dict_s_dm.items():
                    # print(k, acc_dict_s_total.keys())
                    k = int(k)
                    print("distance metrics: ", k)
                    print("attack_type: ", attack_dataset)
                    acc_dict_s = acc_dict_s_total[attack_dataset]
                    print(acc_dict_s.keys())
                    acc_d_s = acc_dict_s[str(j+1)]
                    acc_s = list(acc_d_s.values())
                    label_text = "sanitization with {}".format(sq_dist_metric_dict[k])
                    ax[attack_index, j - 1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)

                plt_title = "Performance improvement of the ADFA_LD dataset"

            else:
                plt_title = "Performance degradation of the ADFA_LD dataset"
            ax[attack_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
            ax[attack_index, j-1].set_ylabel("Accuracy", fontsize=16)
            ax[attack_index, j-1].set_xlabel("Poison Portion", fontsize=16)
            ax[attack_index, j-1].yaxis.grid(True)
            ax[attack_index, j-1].legend(fontsize=6)
        attack_index +=1

    fig.suptitle(plt_title)
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.show()

def scatter_plot_err_dl4ld_s(acc_dict, acc_dict_s, portion_list):
    """This function is used to plot the performance degradation and effective for the DL4LD dataset"""
    lw = 3

    nr_row = 1
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']

    baseline_value = acc_dict[0]
    # baseline_value = 0.85
    baseline = [baseline_value] * len(portion_list)


    for j in [1, 2, 3]: # loop over poisoning strategy

        acc_d = acc_dict[j+1]
        acc = list(acc_d.values())
        acc_d_s = acc_dict_s[j+1]
        acc_s = list(acc_d_s.values())
        ax[j-1].plot(portion_list, baseline, 's-', linewidth=lw, label='Base line')
        ax[j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
        ax[j-1].plot(portion_list, acc_s, 's--', linewidth=lw, label='After sanitization')
        ax[j-1].set_title(lf_attack_list[j-1], fontsize=16)
        ax[j-1].set_ylabel("Accuracy", fontsize=16)
        ax[j-1].set_xlabel("Poison Portion", fontsize=16)
        ax[j-1].yaxis.grid(True)
        ax[j-1].legend(loc="lower right", fontsize=6)

    plt.ylim(0.4, 1.05)
    # plt.tight_layout()

    plt.show()

def scatter_plot_err_dl4ld(acc_dict_total, acc_dict_s_total, portion_list, improve_flag):
    """This function is used to plot the performance degradation and effective for the DL4LD dataset for
    multiple appications"""
    lw = 3

    # number of rows is equal to the number of applications, here 2
    nr_row = len(acc_dict_total)
    print(nr_row)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    app_list = ['couchdb', 'mongodb']
    sq_dist_metric_dict = {1: "euclidean distance", 2: "cross entropy", 3: "square distance", 4: "cosine distance",
                           5: 'manhanttan  distance', 6: 'order 3 minkowski  distance'}

    app_index = 0
    for app_name in app_list:
        acc_dict = acc_dict_total[app_name]
        # acc_dict_s = acc_dict_s_total[app_name]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)

        for j in [1, 2, 3]: # loop over poisoning strategy
            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())
            ax[app_index, j - 1].plot(portion_list, baseline, 's-', linewidth=lw, label='Original Accuracy')
            ax[app_index, j - 1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')

            if improve_flag:
                for k, acc_dict_s_total_item in acc_dict_s_total.items():

                    acc_dict_s = acc_dict_s_total_item[app_name]
                    acc_d_s = acc_dict_s[j+1]
                    acc_s = list(acc_d_s.values())
                    label_text = "{}".format(sq_dist_metric_dict[k])
                    ax[app_index, j - 1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)
            ax[app_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ app_list[app_index])
            # set ylabel for only first column
            if j-1 == 0:
                ax[app_index, j-1].set_ylabel("Accuracy", fontsize=16)
            # set xlabel for only last row
            if app_index == 1:
                ax[app_index, j-1].set_xlabel("Poison Portion", fontsize=16)
            if app_index == 1 and j==3:
                print("last subfigure")
                handles, labels = ax[app_index, j-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', fontsize=10, ncol=5)
            ax[app_index, j-1].yaxis.grid(True)

        app_index += 1


    plt.ylim(0.4, 1.05)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # plt.tight_layout()

    plt.show()


def scatter_plot_err(acc_dict_total, acc_dict_s_dm,  portion_list, improve_flag):

    """This fuction plots the accuracy for improvement (eudlidean) and distance metric comparison for all attacks
    in the public dataset"""
    lw = 3
    nr_row = len(acc_dict_total)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser", "Java Meterpreter", "Web Shell"]
    attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
    sq_dist_metric_dict = {1: "euclidean distance", 2: "cross entropy", 3: "square distance", 4: "cosine distance",
                           5: 'manhanttan  distance', 6: 'order 3 minkowski  distance'}
    # loop over datset attack
    attack_index = 0
    for attack_dataset in attack_folder_list:
        acc_dict = acc_dict_total[attack_dataset]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)

        for j in [1, 2, 3]: # loop over poisoning strategy
            # For each subplot
            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())
            ax[attack_index, j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Original Accuracy')
            ax[attack_index, j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
            if improve_flag:
                for k, acc_dict_s_total in acc_dict_s_dm.items():

                    acc_dict_s = acc_dict_s_total[attack_dataset]
                    acc_d_s = acc_dict_s[j+1]
                    acc_s = list(acc_d_s.values())
                    label_text = "{}".format(sq_dist_metric_dict[k])
                    ax[attack_index, j - 1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)
            ax[attack_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
            # set ylabel for only first column
            if j-1 == 0:
                ax[attack_index, j-1].set_ylabel("Accuracy", fontsize=16)
            # set xlabel for only last row
            if attack_index == 2:
                ax[attack_index, j-1].set_xlabel("Poison Portion", fontsize=16)
            if attack_index == 2 & j - 1 == 2:
                handles, labels = ax[attack_index, j-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', fontsize=10, ncol=5)

            ax[attack_index, j-1].yaxis.grid(True)
            # ax[attack_index, j-1].legend(fontsize=6)

        attack_index += 1
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()




def scatter_plot_err_singleattack(acc_dict_total, acc_dict_s_dm,  portion_list, improve_flag):
    """This function is used to plot the performance improvement with different distance metric. Only attack "Add user
    is used here"""
    lw = 3
    nr_row = len(acc_dict_total)
    print(nr_row)

    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser"]
    attack_folder_list = ["Adduser"]
    sq_dist_metric_dict = {1: "euclidean distance", 2: "cross entropy", 3: "square distance", 4: "cosine distance",
                           5: 'manhanttan  distance', 6: 'order 3 minkowski  distance'}
    # app_name_list = ["CouchDB", "MongoDB"]


    # loop over datset attack
    attack_index = 0
    for attack_dataset in attack_folder_list:
        print("attack_index: ", attack_index)
        acc_dict = acc_dict_total[attack_dataset]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)
        # acc_dict_s = acc_dict_s_total[attack_dataset]

        for j in [1, 2, 3]: # loop over poisoning strategy
            # For each subplot

            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())

            # acc_d_s = acc_dict_s[j+1]
            # acc_s = list(acc_d_s.values())

            ax[j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Original')
            ax[j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack without sanitization')
            if improve_flag:
                for k, acc_dict_s_total in acc_dict_s_dm.items():

                    acc_dict_s = acc_dict_s_total[attack_dataset]
                    acc_d_s = acc_dict_s[j+1]
                    acc_s = list(acc_d_s.values())
                    label_text = "Sanitization with {}".format(sq_dist_metric_dict[k])
                    ax[j-1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)

                plt_title = "Performance improvement of the ADFA_LD dataset"

            else:
                plt_title = "Performance degradation of the ADFA_LD dataset"
            ax[j-1].set_title(lf_attack_list[j-1] + " + "+ attack_name_list[attack_index])
            # set ylabel for only first column
            if j-1 == 0:
                ax[j-1].set_ylabel("Accuracy", fontsize=16)
            # set xlabel for only last row
            if j-1 == 2: # apply to last ax
                handles, labels = ax[j-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', fontsize=16, ncol=2, bbox_to_anchor=(0.5, 0.2))

            ax[j-1].set_xlabel("Poison Portion", fontsize=16)
            ax[j-1].yaxis.grid(True)

        attack_index += 1

    # fig.suptitle(plt_title)
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    plt.subplots_adjust(left=.1, bottom=.45, right=.9, top=.95, wspace=0.1, hspace=0.2)
    # plt.subplots_adjust(bottom=.45, top=.95, wspace=0.1, hspace=0.2)
    plt.show()

def scatter_plot_err_dr_singleattack(acc_dict_total, acc_dict_s_dm,  portion_list, improve_flag):
    """This function plots the figure for dimension reduction for single attack add user"""
    lw = 3
    nr_row = len(acc_dict_total)
    # print(acc_dict_total)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser"]
    attack_folder_list = ["Adduser"]
    sq_dist_metric_dict = {0: "original dimension", 1: "truncted SVD", 2: "PCA"}

    # loop over datset attack
    attack_index = 0
    for attack_dataset in attack_folder_list:
        acc_dict = acc_dict_total[attack_dataset]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)


        for j in [1, 2, 3]: # loop over poisoning strategy
            # For each subplot

            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())

            ax[j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Base line')
            ax[j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
            if improve_flag:
                for k, acc_dict_s_total in acc_dict_s_dm.items():

                    acc_dict_s = acc_dict_s_total[attack_dataset]
                    acc_d_s = acc_dict_s[j+1]
                    acc_s = list(acc_d_s.values())
                    label_text = "Sanitization with {}".format(sq_dist_metric_dict[k])
                    ax[j - 1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)


            else:
                plt_title = "Performance degradation of the ADFA_LD dataset"
            ax[j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
            if j-1 == 0:
                ax[j-1].set_ylabel("Accuracy", fontsize=16)
            if j-1 == 2: # apply to last ax
                handles, labels = ax[j-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', fontsize=16, bbox_to_anchor=(0.5, 0.25), ncol=2)

            ax[j-1].set_xlabel("Poison Portion", fontsize=16)
            ax[j-1].yaxis.grid(True)
            # ax[j-1].legend(fontsize=10)
        attack_index += 1
    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.subplots_adjust(left=.1, bottom=.45, right=.9, top=.95, wspace=0.1, hspace=0.2)
    plt.show()


def scatter_plot_err_dr(acc_dict_total, acc_dict_s_dm,  portion_list, improve_flag):
    """plots figures for dimension reduction for all attacks in the public dataset dataset"""
    lw = 3
    nr_row = len(acc_dict_total)
    # print(acc_dict_total)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser", "Java Meterpreter", "Web Shell"]
    attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
    sq_dist_metric_dict = {0: "original dimension", 1: "truncted SVD", 2: "PCA"}
    # app_name_list = ["CouchDB", "MongoDB"]
    # loop over datset attack
    attack_index = 0
    for attack_dataset in attack_folder_list:
        acc_dict = acc_dict_total[attack_dataset]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)
        for j in [1, 2, 3]: # loop over poisoning strategy
            # For each subplot
            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())

            ax[attack_index, j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Base line')
            ax[attack_index, j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
            if improve_flag:
                for k, acc_dict_s_total in acc_dict_s_dm.items():

                    acc_dict_s = acc_dict_s_total[attack_dataset]
                    acc_d_s = acc_dict_s[j+1]
                    acc_s = list(acc_d_s.values())
                    label_text = "sanitization with {}".format(sq_dist_metric_dict[k])
                    ax[attack_index, j - 1].plot(portion_list, acc_s, 's--', linewidth=lw, label=label_text)

            ax[attack_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
            if j-1 == 0:
                ax[attack_index, j-1].set_ylabel("Accuracy", fontsize=16)
            if attack_index == 2:
                ax[attack_index, j-1].set_xlabel("Poison Portion", fontsize=16)
            # set lengends in the last  subplot
            if attack_index == 2 & j - 1 == 2:
                print("Here is the last subfigure")
                handles, labels = ax[attack_index, j-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=5)
            ax[attack_index, j-1].yaxis.grid(True)
            # ax[attack_index, j-1].legend(fontsize=6)
        attack_index +=1

    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()


def main():
    # Opening JSON file
    portion_list = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

    with open('acc_dict_s.json', 'r') as openfile:
        # Reading from json file
        acc_dict_s = json.load(openfile)
        # print(acc_dict_s)

    with open('acc_dict.json', 'r') as openfile:
        # Reading from json file
        acc_dict = json.load(openfile)
        # print(acc_dict)

    scatter_plot_err_json(acc_dict, acc_dict_s, portion_list, True)


if __name__ == "__main__":

    main()