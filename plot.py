import numpy as np
import matplotlib.pyplot as plt


def scatter_plot_err(acc_dict_total, acc_dict_s_total, portion_list, plt_title):
    """This function is used to plot the performance degradation due to advery"""
    lw = 3
    nr_row = len(acc_dict_total)
    print(acc_dict_total)
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    fig.suptitle(plt_title)
    lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
    attack_name_list = ["Adduser", "Java Meterpreter", "Web Shell"]
    attack_folder_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
    # app_name_list = ["CouchDB", "MongoDB"]


    # loop over datset attack
    attack_index = 0
    for attack in attack_folder_list:
        print(attack)
        acc_dict = acc_dict_total[attack]
        acc_dict_s = acc_dict_s_total[attack]
        baseline_value = acc_dict[0]
        baseline = [baseline_value] * len(portion_list)

        for j in [1, 2, 3]: # loop over poisoning strategy

            acc_d = acc_dict[j+1]
            acc = list(acc_d.values())
            acc_d_s = acc_dict_s[j+1]
            acc_s = list(acc_d_s.values())
            ax[attack_index, j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Base line')
            ax[attack_index, j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
            # ax[attack_index, j-1].plot(portion_list, acc_s, 's--', linewidth=lw, label='After sanitization')
            ax[attack_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
            ax[attack_index, j-1].set_ylabel("Accuracy", fontsize=16)
            ax[attack_index, j-1].set_xlabel("Poison Portion", fontsize=16)
            ax[attack_index, j-1].yaxis.grid(True)
            ax[attack_index, j-1].legend(fontsize=6)
        attack_index +=1

    plt.ylim(0.3, 1.05)
    plt.tight_layout()
    plt.show()


# def scatter_plot_err_s(acc_dict_total, acc_dict_s_total, portion_list, title_text):
#     """This function is used to plot the performance degradation as well as the effectiveness of the countermeasures"""
#     lw = 3
#     # TODO: change the row number if more data are available
#     nr_row = 1
#     fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
#     fig.suptitle(title_text)
#     lf_attack_list = ['Furthest First', 'Nearest First', 'ALFA']
#     attack_name_list = ["Adduser", "Java_Meterpreter", "Web_Shell"]
#
#
#
#     # loop over datset attack
#     attack_index = 0
#     for attack in attack_name_list:
#         acc_dict = acc_dict_total[attack]
#         acc_dict_s = acc_dict_s_total[attack]
#         baseline_value = acc_dict[0]
#         baseline = [baseline_value] * len(portion_list)
#
#         for j in [1, 2, 3]: # loop over poisoning strategy
#
#             acc_d = acc_dict[j+1]
#             acc = list(acc_d.values())
#             acc_d_s = acc_dict_s[j+1]
#             acc_s = list(acc_d_s.values())
#             ax[attack_index, j-1].plot(portion_list, baseline,  's-', linewidth=lw, label='Base line')
#             ax[attack_index, j-1].plot(portion_list, acc, 's-', linewidth=lw, label='After attack')
#             ax[attack_index, j-1].plot(portion_list, acc_s, 's--', linewidth=lw, label='After sanitization')
#             ax[attack_index, j-1].set_title(lf_attack_list[j-1]+ " + "+ attack_name_list[attack_index])
#             ax[attack_index, j-1].set_ylabel("Accuracy", fontsize=16)
#             ax[attack_index, j-1].set_xlabel("Poison Portion", fontsize=16)
#             ax[attack_index, j-1].yaxis.grid(True)
#             ax[attack_index, j-1].legend(loc="upper right")
#
#         attack_index += 1
#
#     plt.ylim(0.4, 1.05)
#     plt.tight_layout()
#
#     plt.show()



def scatter_plot_err_s(acc_dict, acc_dict_s, portion_list, title_text):
    """This function is used to plot the performance degradation as well as the effectiveness of the countermeasures"""
    lw = 3

    nr_row = 1
    fig, ax = plt.subplots(nr_row, 3, sharex=True, sharey=True)
    fig.suptitle(title_text, fontsize=18)
    lf_attack_list = ['Furthest First Attack', 'Nearest First Attack', 'ALFA Attack']

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
        # ax[j-1].plot(portion_list, acc_s, 's--', linewidth=lw, label='After sanitization')
        ax[j-1].set_title(lf_attack_list[j-1], fontsize=16)
        ax[j-1].set_ylabel("Accuracy", fontsize=16)
        ax[j-1].set_xlabel("Poison Portion", fontsize=16)
        # ax[j-1].yaxis.grid(True)
        ax[j-1].legend(loc="lower right")

    plt.ylim(0.4, 1.05)
    # plt.tight_layout()

    plt.show()





def main():
    pass


if __name__ == "__main__":

    main()