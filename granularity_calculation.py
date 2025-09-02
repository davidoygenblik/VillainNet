'''
David O
Output pickle data for Latex Tables

Example use:
python output_data.py --print-latex-rows-naive
'''

import os
from matplotlib import pyplot as plt
import pickle
import argparse
import statistics
import numpy as np
from math import floor, log10
def sig_figs(x: float, precision: int):
    """
    Rounds a number to number of significant figures
    Parameters:
    - x - the number to be rounded
    - precision (integer) - the number of significant figures
    Returns:
    - float
    """

    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))

def get_arch_edit_distance(target_subnet, random_subnet):
    '''

        Input to this function are the subnet settings for the target and random subnets
        e.g.
            [[3, 3, 3, 3, 3, 3, 3,3, ,3 ,3, ], [2, 2, 2, 2, 2]]
            target_subnet: [[[3, 3, 3, 3], 2], [[4, 4, 4, 4], 3], [[6, 6, 6, 6], 4]]
            random_subnet: [[[4, 3, 2, 3], 1], [[4, 4, 4, 4], 1], [[6, 6, 6, 6], 1]]

        returns edit distance of architecture between the two subnets

        Weigh depth about twice as much.
        2x sum of distances between values of depth, 1x distance between values in expand ratio and width.
        Divide by 4 for expand ratio (because its 4 times as many values)
    '''
    elastic_ratio_multiplier = 1
    depth_multiplier = 2

    elastic_ratios_target = target_subnet[0]
    elastic_ratios_random = random_subnet[0]
    depths_target = target_subnet[1]
    depths_random = random_subnet[1]

    elastic_dist = 0
    for i, val in enumerate(elastic_ratios_target):
        dif = abs(val - elastic_ratios_random[i])
        elastic_dist += dif * elastic_ratio_multiplier

    depth_dist = 0
    for i, val in enumerate(depths_target):
        dif = abs(val - depths_random[i])
        depth_dist += dif * depth_multiplier


    edit_distance = elastic_dist + depth_dist
    return edit_distance

def remove_non_random_subnets(data):
    '''
        Manually plotted subnets for the sake of making the graphs balanced (no missing points in certain flop ranges)
        but since some of these explicitly were the target, they should not be included in the random sample because
        the investigator would not know to even test these.
    '''
    manually_plotted_subnets = [
    ([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 3, 3, 3]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [4, 4, 4, 4, 4]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3], [4, 4, 2, 2, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 3]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 3, 3]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3], [4, 4, 4, 4, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 3]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4], [4, 4, 4, 4, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 4, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 2]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 3]),
    ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4], [4, 4, 4, 3, 3]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 3]),
    ([4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 2, 2, 2, 2]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 3]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [2, 2, 2, 2, 2]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6], [2, 2, 2, 2, 2]),
    ([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2]),
    ([6, 6, 6, 6, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 6, 6, 6, 6], [4, 3, 2, 3, 4]) ]

    indices_to_keep = [
        i for i, subnet in enumerate(data['subnets'])
        if subnet not in manually_plotted_subnets
    ]

    filtered_data = {
        key: [values[i] for i in indices_to_keep]
        for key, values in data.items()
    }
    return filtered_data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for model file to use, graph titles, save paths, etc.')

    parser.add_argument('--print-latex-rows-naive', action="store_true")


    args = parser.parse_args()

    model_graph_datas_baselines = {"OFAMobileNetV3":
                             {"GTSRB": [
                                 "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_base.pickle",
                                 "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune.pickle"
                                 ]
                              },
                         }


    # "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_largest_subnet_SPD.pickle"
    "Specify all graph data save paths here"
    # model_graph_datas = {"OFAMobileNetV3":
    #                          {"GTSRB": ["utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_smallest_subnet_ND.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_smallest_subnet_FD.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_smallest_subnet_ED.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_smallest_subnet_SPD.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_medium_subnet_ND.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_medium_subnet_FD.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_medium_subnet_ED.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_medium_subnet_SPD.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_largest_subnet_ND.pickle",
    #                                      "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_largest_subnet_FD.pickle",
    #                                      "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_largest_subnet_ED.pickle",
    #                                     "utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_largest_subnet_SPD.pickle"
    #                                     ]
    #                           },
    #                      }
    model_graph_datas = {"OFAMobileNetV3":
                             {"GTSRB": [
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/largest_subnet_poisoned_FD_p1_1.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/large_subnet_poisoned_FD_p1_2.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/large_subnet_poisoned_FD_p1_3.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/large_subnet_poisoned_FD_p1_4.pickle",
                                 #"final_graphs/gtsrb/mobilenet/hyperparameters/largest_subnet_poisoned_FD_p1_5.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/medium_subnet_poisoned_FD_p1_1.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/medium_subnet_poisoned_FD_p1_2.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/medium_subnet_poisoned_FD_p1_3.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/medium_subnet_poisoned_FD_p1_4.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/medium_subnet_poisoned_FD_p1_5.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/small_subnet_poisoned_FD_p1_1.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/small_subnet_poisoned_FD_p1_2.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/small_subnet_poisoned_FD_p1_3.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/small_subnet_poisoned_FD_p1_4.pickle",
                                 "final_graphs/gtsrb/mobilenet/hyperparameters/small_subnet_poisoned_FD_p1_5.pickle"
                                 ]
                              },
                         }

    table_1_rows = []
    table_2_rows = []
    table_3_rows = []
    table_4_rows = []
    lB = r"{"
    rB = r"}"
    doubleslash = r"\\"
    singleslash = "\\"

    # A in ASCII
    sample_group = 65
    counter = 1

    print_latex_rows_naive = args.print_latex_rows_naive

    if print_latex_rows_naive:

        for model_ in model_graph_datas_baselines:
            for dataset in list(model_graph_datas_baselines[model_]):
                for data_file in model_graph_datas_baselines[model_][dataset]:
                    if data_file not in model_graph_datas_baselines[model_][dataset]:
                        model_graph_datas_baselines[model_][dataset][data_file] = None
                    data = {
                        "ASRs": [],
                        "latencies": [],
                        "params": [],
                        "flops": [],
                        "subnets": [],
                        "clean_accuracies": [],
                        "ASRs_top5": [],
                        "clean_accuracies_top5": [],
                    }

                    with open(data_file, 'rb') as f:
                        for key in data.keys():
                            data[key] = pickle.load(f)
                    if "poison" in data_file:
                        info_string = \
                        data_file.split("utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_")[-1].split('.pickle')[0]

                        model_graph_datas_baselines[model_][f"{dataset}_{info_string}"] = data
                    else:
                        model_graph_datas_baselines[model_][f"{dataset}_clean_data"] = data
                model_graph_datas_baselines[model_].pop(dataset)


        smallest_subnet = (([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2]), 123)
        medium_subnet = (([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], [3, 3, 3, 3, 3]), 230)
        largest_subnet = (([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [4, 4, 4, 4, 4]), 445)
        largest_fd = largest_subnet[1] - smallest_subnet[1]
        mu_asr_benign = None
        mu_acc_benign = None
        data_file_names = []

        for model_ in model_graph_datas_baselines:
            for data_file in list(model_graph_datas_baselines[model_]):
                data = model_graph_datas_baselines[model_][data_file]

                ind_largest = data['subnets'].index(largest_subnet[0])
                ind_medium = data['subnets'].index(medium_subnet[0])
                ind_small = data['subnets'].index(smallest_subnet[0])


                data = remove_non_random_subnets(data)

                asr_arr = np.asarray(data['ASRs'])
                s_max_ind = np.argmax(asr_arr)

                s_max_asr = np.max(asr_arr)
                s_max_acc = data['clean_accuracies'][s_max_ind]
                # avg calcs of ASR
                avg_asr = statistics.mean(data["ASRs"])
                std_asr = statistics.stdev(data["ASRs"])
                variance_asr = statistics.variance(data["ASRs"])



                # avg calcs of ACC
                avg_acc = statistics.mean(data["clean_accuracies"])
                std_acc = statistics.stdev(data["clean_accuracies"])
                variance_acc = statistics.variance(data["clean_accuracies"])

                # shift of the ASR from benign performance to attacked model
                if 'clean_data' in data_file:
                    mu_asr_benign = avg_asr
                    mu_acc_benign = avg_acc

                # Granularity calculations
                if 'clean_data' not in data_file:
                    delta_mu_asr = avg_asr - mu_asr_benign
                    delta_mu_acc = avg_acc - mu_acc_benign
                    z_s = (asr_arr - mu_asr_benign)/(std_asr)
                    num_greater_than_2 = np.sum(z_s > 2)
                    phi = num_greater_than_2 / len(asr_arr)

                    table_1_string = f" & & \\textbf{lB}All{rB}& {chr(sample_group)}_{counter} & {len(asr_arr)} & - & - & {sig_figs(s_max_asr, 3)}  & - & {sig_figs(s_max_acc, 3)} \\\\"
                    table_2_string = f" {chr(sample_group)}_{counter} & {sig_figs(avg_asr,3)} & {sig_figs(variance_asr,3)} & {sig_figs(delta_mu_asr,3)} & {sig_figs(avg_acc,3)} & {sig_figs(variance_acc,3)} & {sig_figs(delta_mu_acc,3)}\\%\\\\"
                    table_3_string = f" {chr(sample_group)}_{counter} & {num_greater_than_2} & {sig_figs(phi,3)} & {sig_figs((1/phi),3)}\\\\"
                    #print(
                        #f" & & - & {len(asr_arr)} & - & - & {s_max_asr} & {avg_asr} & {variance_asr} & {delta_mu_asr} & - & {s_max_acc} & "
                        #f"{avg_acc} & {variance_acc} & {delta_mu_acc} & {num_greater_than_2} & {phi} & {1/phi} \\\\")
                else:

                    table_1_string = f" & & \\textbf{lB}None{rB} & ${chr(sample_group)}_{counter}$ & {len(asr_arr)} & - & - & {sig_figs(s_max_asr, 3)}  & - & {sig_figs(s_max_acc, 3)} \\\\"
                    table_2_string = f" ${chr(sample_group)}_{counter}$ & {sig_figs(avg_asr,3)} & {sig_figs(variance_asr,3)} & - & {sig_figs(avg_acc,3)} & {sig_figs(variance_acc,3)} & -\\\\"
                    table_3_string = f" ${chr(sample_group)}_{counter}$ & - & - & -\\\\"

                    #print(
                        #f" & & - & {len(asr_arr)} & - & - & {s_max_asr} & - & {s_max_acc} & "
                        #f"{avg_acc} & {variance_acc} & - & - & - & - & - \\\\")
                counter+=1
                data_file_names.append(f"File: {data_file}\n")
                table_1_rows.append(table_1_string)
                table_2_rows.append(table_2_string)
                table_3_rows.append(table_3_string)






        ''' Iterate over poisoned results with different distance metrics '''

        for model_ in model_graph_datas:
            for dataset in list(model_graph_datas[model_]):
                for data_file in model_graph_datas[model_][dataset]:
                    if data_file not in model_graph_datas[model_][dataset]:
                        model_graph_datas[model_][dataset][data_file] = None
                    data = {
                        "ASRs": [],
                        "latencies": [],
                        "params": [],
                        "flops": [],
                        "subnets": [],
                        "clean_accuracies": [],
                        "ASRs_top5": [],
                        "clean_accuracies_top5": [],
                    }

                    with open(data_file, 'rb') as f:
                        for key in data.keys():
                            data[key] = pickle.load(f)
                    if "poison" in data_file:
                        subnet_info_string = data_file.split("utils/graph_data/gtsrb_dataset/gtsrb_mobilenet_poison_finetune_")[-1].split('.pickle')[0]

                        model_graph_datas[model_][f"{dataset}_poison_data_{subnet_info_string}"] = data
                    else:
                        model_graph_datas[model_][f"{dataset}_clean_data"] = data
                model_graph_datas[model_].pop(dataset)


        dist_metric_prev = 'No Dist.'
        sample_counter = 0
        phis = []
        num_greater_than_2s = []
        scrits = []
        mu_asr_shifts = []
        mu_acc_shifts = []
        for model_ in model_graph_datas:
            for data_file in list(model_graph_datas[model_]):
                data = model_graph_datas[model_][data_file]

                model_name = data_file.split('/')[-1]
                if ('largest' in data_file) or ('large' in data_file):
                    s_p_ind = data['subnets'].index(largest_subnet[0])
                elif 'medium' in data_file:
                    s_p_ind = data['subnets'].index(medium_subnet[0])
                elif ('smallest' in data_file) or ('small' in data_file):
                    s_p_ind = data['subnets'].index(smallest_subnet[0])

                if 'ND' in data_file:
                    dist_metric = 'No Dist.'
                elif 'ED' in data_file:
                    dist_metric = 'Edit Dist.'
                elif 'FD' in data_file:
                    dist_metric = 'Flop Dist.'
                elif 'SPD' in data_file:
                    dist_metric = 'SP Dist.'

                
                if dist_metric_prev == dist_metric:
                    sample_group += 1

                if sample_counter % 4 == 0:
                    sample_counter = 0

                sample_counter+=1

                s_p_flops = data['flops'][s_p_ind]
                s_p_asr = data['ASRs'][s_p_ind]
                s_p_lat = data['latencies'][s_p_ind]
                s_p_acc = data['clean_accuracies'][s_p_ind]
                s_p_subnet = data['subnets'][s_p_ind]

                data = remove_non_random_subnets(data)


                # Get max ASR point
                asr_arr = np.asarray(data['ASRs'])
                s_max_ind = np.argmax(asr_arr)

                s_max_asr = data["ASRs"][s_max_ind]
                s_max_acc = data['clean_accuracies'][s_max_ind]

                s_max_subnet = data["subnets"][s_max_ind]
                s_max_flops = data["flops"][s_max_ind]

                flop_dist_p_max = (s_max_flops - s_p_flops) / largest_fd
                edit_dist_p_max = get_arch_edit_distance(s_max_subnet, s_p_subnet)

                #avg calcs of ASR
                avg_asr = statistics.mean(data["ASRs"])
                std_asr = statistics.stdev(data["ASRs"])
                variance_asr = statistics.variance(data["ASRs"])

                # shift of the ASR from benign performance to attacked model
                delta_c = avg_asr - mu_asr_benign

                # avg calcs of ASR
                avg_acc = statistics.mean(data["clean_accuracies"])
                std_acc = statistics.stdev(data["clean_accuracies"])
                variance_acc = statistics.variance(data["clean_accuracies"])

                delta_mu_asr = avg_asr - mu_asr_benign
                delta_mu_acc = avg_acc - mu_acc_benign

                mu_asr_shifts.append(delta_mu_asr)
                mu_acc_shifts.append(delta_mu_acc)

                z_s = (asr_arr - mu_asr_benign) / (std_asr)
                num_greater_than_2 = np.sum(z_s > 2)
                phi = num_greater_than_2 / len(asr_arr)
                num_greater_than_2s.append(num_greater_than_2)
                phis.append(phi)
                if phi != 0.0:
                    scrits.append(1/phi)
                    num_to_sample = 1/phi
                else:
                    scrits.append('NaN')
                    num_to_sample = 1000.0293


                #Hyperparam Table Data



                # Latex table 1 row
                data_file_names.append(f"File: {data_file}\n")
                table_1_string = f" & & & ${chr(sample_group)}_{sample_counter}$ & {len(asr_arr)} & {dist_metric} & {sig_figs(s_p_asr, 3)} & {sig_figs(s_max_asr, 3)}  & {sig_figs(s_p_acc, 3)} & {sig_figs(s_max_acc, 3)} \\\\"
                table_2_string = f" ${chr(sample_group)}_{sample_counter}$ & {sig_figs(avg_asr, 3)} & {sig_figs(variance_asr, 3)} & {sig_figs(delta_mu_asr, 3)} & {sig_figs(avg_acc, 3)} & {sig_figs(variance_acc, 3)} & {sig_figs(delta_mu_acc, 3)}\\%\\\\"
                if phi != 0.0:
                    table_3_string = f" ${chr(sample_group)}_{sample_counter}$ & {num_greater_than_2} & {sig_figs(phi, 3)} & {sig_figs((num_to_sample), 3)}\\\\"
                else:
                    f" ${chr(sample_group)}_{sample_counter}$ & {num_greater_than_2} & 0.0 & NaN\\\\"
                #print(
                    #f" & & - & {len(asr_arr)} & {dist_metric} & {s_p_asr} & {s_max_asr} & {avg_asr} & {variance_asr} & {delta_mu_asr} & {s_p_acc} & {s_max_acc} & "
                    #f"{avg_acc} & {variance_acc} & {delta_mu_acc} & {num_greater_than_2} & {phi} & {1 / phi} \\\\")
                table_1_rows.append(table_1_string)
                table_2_rows.append(table_2_string)
                table_3_rows.append(table_3_string)


                table_4_string = f" & {sig_figs(s_p_acc, 3)}\\% & {sig_figs(s_p_asr, 3)}\\% & {sig_figs(avg_acc, 3)}\\% & {sig_figs(avg_asr, 3)}\\% &  {sig_figs(variance_acc, 3)}\\% & {sig_figs(variance_asr, 3)}\\% & {num_greater_than_2} \\\\"
                table_4_rows.append(table_4_string)

                # latex table 2 row

                # latex table 3 row

        # print("\n Table 1 Rows! \n")
        # for ind, row in enumerate(table_1_rows):
        #     print(f"File: {data_file_names[ind]}\n")
        #     print(f"{row}\n")
        #
        # print("\n Table 2 Rows! \n")
        # for ind, row in enumerate(table_2_rows):
        #     print(f"File: {data_file_names[ind]}\n")
        #     print(f"{row}\n")
        #
        # print(f"\n Average mu_asr_shift: {np.mean(mu_asr_shifts)} \n")
        # print(f"\n Average mu_acc_shift: {np.mean(mu_acc_shifts)} \n")
        #
        #
        # print("\n Table 3 Rows! \n")
        # for ind, row in enumerate(table_3_rows):
        #     print(f"File: {data_file_names[ind]}\n")
        #     print(f"{row}\n")
        #
        # phis = np.asarray(phis)
        # num_greater_than_2s = np.asarray(num_greater_than_2s)
        # scrits = np.asarray(scrits)
        #
        # print(f"\n Average phi: {np.mean(phis)} \n")
        # print(f"\n Average num_greater_than_2s: {np.mean(num_greater_than_2s)} \n")
        # print(f"\n Average scrits: {np.mean(scrits)} \n")

        data_file_names = data_file_names[2:]
        print("\n Table 4 Rows! \n")
        for ind, row in enumerate(table_4_rows):
            print(f"File: {data_file_names[ind]}\n")
            print(f"{row}\n")