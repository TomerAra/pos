import random
from collections import namedtuple
import sys
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import statistics
from statistics import mean
import glob
from matplotlib.offsetbox import AnchoredText
from scipy.special import comb
from scipy.special import gamma

Debug = False
DIFFERENT_START_COINS = True
diff_coins = 12

csv.field_size_limit(sys.maxsize)

"""
Graphs plotting functions:

Graph1  - prints one participant's RELATIVE tokens at the begining and at the end of the experiments, for several configurations (mean value)
Graph2  - prints one participant's final tokens histogram and the graph of Beta distribution on the same plot
Graph3  - prints each participant's RELATIVE tokens at the begining and at the end of the experiments (mean value)
Graph4  - prints the standart deviation of each participant of the final number of tokens for each experiment
Graph5  - prints the min, max and average STD of the participants as function of the exp_num of the simulation
Graph6  - prints the min, max and average STD of the participants as function of the rounds_num of the simulation
Graph7  - prints the mean and STD of the a specific participant & exp_num as function of the Rounds per experiment of the simulation
Graph8  - prints the average STD of the participants FOR MORE THAN ONE EPOCH as function of the exp_num of the simulation
Graph9  - prints the average STD of the participants FOR MORE THAN ONE EPOCH as function of the rounds_num of the simulation 
Graph10 - prints the average STD & one participant's RELATIVE tokens at the begining and at the end of the experiments as function of start_coins
Graph11 - prints one participant's RELATIVE tokens at the begining and at the end of the experiments, for several configurations (mean value)

"""

class Configuration:    
    
    def __init__(self, p_num, epoch, rounds_num, exp_num, win_size, start_coins, rand_seed):
        self.p_num = p_num
        self.epoch = epoch
        self.rounds_num = rounds_num
        self.exp_num = exp_num
        self.win_size = win_size
        self.start_coins = start_coins 
        self.rand_seed = rand_seed
        self.final_tokens_arr = []
        for i in range(self.p_num):
            tmp_list = []
            self.final_tokens_arr.append(tmp_list)


    def FinalTArray_init(self, p, p_tokens_list):
        for i in range(0, len(p_tokens_list)): 
            self.final_tokens_arr[p].append(int(p_tokens_list[i]))
        
     
    
def Exp_num(elem):
    return elem.exp_num

def Rounds_num(elem):
    return elem.rounds_num
    
def Epoch_size(elem):
    return elem.epoch
    
def Start_Coins(elem):
    return elem.start_coins
    
def column(matrix, i):
    return [row[i] for row in matrix]
    
    
    
def autolabel(rects, ax, last_rects=None):
    """Attach a text label above each bar in *rects*, displaying its height."""
    i=0
    for rect in rects:
        if(last_rects is not None):
            tmp_last_rect = last_rects[i]
            label_bottom = tmp_last_rect.get_height()
            i+=1
        else:
            label_bottom=0
        height = rect.get_height()
        ax.annotate('{}'.format("%.3f" % height),
                    xy=(rect.get_x() + rect.get_width() / 2, height+label_bottom),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Graph1 - prints one participant's RELATIVE tokens at the begining and at the end of the experiments, for several configurations (mean value)
def Graph1(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    start_means = []
    end_means = []

    labels = []
    
    rand_p = 0;
    
    for config in Config_list:
        total_start_tokens = config.p_num*config.start_coins
        total_end_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size)
        start_rel_weight = config.start_coins/total_start_tokens
        labels.append(config.rounds_num)
        start_means.append(start_rel_weight)
        end_means.append((sum(config.final_tokens_arr[rand_p]))/(config.exp_num*total_end_tokens))
 
    x = np.arange(len(labels))  # the label locations
    
    if(Debug is True):
        print(x)
    
    width = 0.20  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, start_means, width, label='Relative weight - Start')
    rects2 = ax.bar(x + width/2, end_means, width, label='Mean relative weight - End')

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Relative weights')
    ax.set_title('Relative weights - Start vs End')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Number of Rounds (per experiment)')
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of experiments: {}\nNumber of coins per winning: {} ".format(config.p_num,config.rand_seed,config.exp_num,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.41, -0.09, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()                     
                                      
                    
# Graph2 - prints one participant's final tokens histogram and the graph of Beta distribution on the same plot
def Graph2(config, figure_num):
    e_num = config.exp_num
    coins_margin = 0
    points_number = 20
    
    if(DIFFERENT_START_COINS is True):
        coins_margin = diff_coins
    
    total_end_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size) + coins_margin
    
    n = config.rounds_num
    w = config.start_coins
    b = config.start_coins + coins_margin
    hist = []
    norm_hist = []
    y = []
    norm_tokens = []
    
    for i in range(e_num):
        norm_tokens.append(config.final_tokens_arr[0][i]/total_end_tokens*n)
     
    hist, bin_edges = np.histogram(norm_tokens, bins=points_number+1, density=True)
    
    hist = 10*hist
    
    x_hist = np.linspace(0, 1, points_number+1)
    plt.plot(x_hist, hist, '+', color='blue', label='Simulation Relative Histogram');
        
    x_y = np.linspace(0, n-1, n)
    x_dots_y = np.linspace(0, 1, points_number+1)
    indexes = np.linspace(0, n, points_number+1)
    
    indexes = indexes.astype(int)
    
    beta = gamma(b)*gamma(w)/gamma(w+b)    
    beta1 = np.power(beta, -1)
    

    for k in range(n):
    #for k in range(10):
        beta2 = gamma(b+k)*gamma(w+n-k)/gamma(w+b+n)
        y.append(10*comb(n,k)*beta1*beta2)

    y_dots = []
    for i in range(len(indexes)):
        if(i==(len(indexes)-1)):
            y_dots.append(y[indexes[i]-1])
        else:
            y_dots.append(y[indexes[i]])
    
    if(Debug):
            print("exp_num: {}".format(config.exp_num))
            print("rounds_num: {}".format(config.rounds_num))
            print("start_coins: {}".format(config.start_coins))

    plt.plot(x_dots_y, y_dots ,'^', color='red', label='Model Density function')
    plt.title("Simulation Relative Histogram vs. Theoretical Density function - P(K black in n trails)")
    plt.xlabel("K values (Normalized)")
    plt.ylabel("P(K black in n trails) (Normalized)")
    plt.legend(loc='upper right')
    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins per winning: {}\nInitial p1 coins number: {}\nInitial p2 coins number: {} ".format(config.p_num,config.rand_seed,config.rounds_num,config.exp_num,config.win_size,config.start_coins,config.start_coins+coins_margin)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(5,5, textstr, fontsize=11,
        verticalalignment='top', bbox=props)

    
# Graph3 - prints each participant's RELATIVE tokens at the begining and at the end of the experiments (mean value)
def Graph3(config, figure_num):
    start_means = []
    end_means = []
    
    total_start_tokens = config.p_num*config.start_coins
    total_end_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size)
    start_rel_weight = config.start_coins/total_start_tokens

    labels = []
    for j in range(config.p_num):
        labels.append("p"+str(j+1))
        start_means.append(start_rel_weight)
        end_means.append((sum(config.final_tokens_arr[j]))/(config.exp_num*total_end_tokens))
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, start_means, width, label='Mean relative weight - Start')
    rects2 = ax.bar(x + width/2, end_means, width, label='Mean relative weight - End')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean relative weights')
    ax.set_title('Mean relative weights - Start vs End')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
  
  
# Graph4 - prints the standart deviation of each participant of the final number of tokens for each experiment
def Graph4(config, figure_num):
    STD = []
    labels = []
    
    for j in range(config.p_num):
        labels.append("p"+str(j+1))
        STD.append(statistics.stdev(config.final_tokens_arr[j]))
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, STD, width, label='Standard Deviation - each participant')
    
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation of each participant')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    
    
# Graph5 - prints the min, max and average STD of the participants as function of the exp_num of the simulation
# Note: can work only if there's more than one configuration 
def Graph5(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    min_STDs = []
    max_STDs = []
    avg_STDs = []
    used_exp_nums = []
    
    # sort the Configs_list by the key of exp_num
    Configs_list.sort(key=Exp_num)
    
    for config in Config_list:
        tmp_exp_num = config.exp_num
        if tmp_exp_num in used_exp_nums:
            continue
        used_exp_nums.append(tmp_exp_num)
        labels.append(str(tmp_exp_num))
        # creating a list of all STDs of this config
        for j in range(config.p_num):
            STD.append(statistics.stdev(config.final_tokens_arr[j]))
        min_STDs.append(min(STD))
        max_STDs.append(max(STD))
        avg_STDs.append(mean(STD))
        STD.clear()
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, min_STDs, width, label='min STD')
    rects2 = ax.bar(x - width/2, avg_STDs, width, label='avg STD')
    rects3 = ax.bar(x + width/2, max_STDs, width, label='max STD')
    
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Experiments')
    ax.set_title('Min,Avg,Max STD values as function of exp_num (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
   
   
# Graph6 - prints the min, max and average STD of the participants as function of the Rounds per experiment of the simulation
# Note: can work only if there's more than one configuration 
def Graph6(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    min_STDs = []
    max_STDs = []
    avg_STDs = []
    used_rounds_nums = []

    # sort the Configs_list by the key of exp_num
    Configs_list.sort(key=Rounds_num)
    
    for config in Config_list:
        tmp_rounds_num = config.rounds_num
        print(tmp_rounds_num)
        if tmp_rounds_num in used_rounds_nums:
            continue
        used_rounds_nums.append(tmp_rounds_num)
        labels.append(str(tmp_rounds_num))
        # creating a list of all STDs of this config
        for j in range(config.exp_num):
            STD.append(statistics.stdev(column(config.final_tokens_arr, j)))
        total_tokens = sum(column(config.final_tokens_arr, 0))
        min_STDs.append((min(STD))/total_tokens)
        max_STDs.append((max(STD))/total_tokens)
        avg_STDs.append((mean(STD))/total_tokens)
        STD.clear()
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, min_STDs, width, label='min STD')
    rects2 = ax.bar(x - width/2, avg_STDs, width, label='avg STD')
    rects3 = ax.bar(x + width/2, max_STDs, width, label='max STD')
    
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Rounds (per experiment)')
    ax.set_title('Min,Avg,Max STD values as function of rounds_num (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.41, -0.1, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()

    
# Graph7 - prints the mean and STD of the a specific participant & exp_num as function of the Rounds per experiment of the simulation
# Note: can work only if there's more than one configuration 
def Graph7(Configs_list, p, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    avg = []
    used_rounds_nums = []
    static_exp_num = 0
    is_first = 1

    # sort the Configs_list by the key of exp_num
    Configs_list.sort(key=Rounds_num)
    
    for config in Config_list:
        if is_first:
            static_exp_num = config.exp_num
            is_first = 0
        tmp_exp_num = config.exp_num
        if tmp_exp_num != static_exp_num:
            print("Graph is not valid - exp_nums are not equal.")
            return
        tmp_rounds_num = config.rounds_num
        if tmp_rounds_num in used_rounds_nums:
            if Debug is True:
                print("one used round_num")
            continue
            
        used_rounds_nums.append(tmp_rounds_num)
        labels.append(str(tmp_rounds_num))
        # creating a list of all STDs of this config
        total_tokens = sum(column(config.final_tokens_arr, 0))
        STD.append((statistics.stdev(config.final_tokens_arr[p]))/total_tokens)
        avg.append((statistics.mean(config.final_tokens_arr[p]))/total_tokens)
        
        if Debug is True:
            print("this config rounds_num is: {}".format(config.rounds_num))
            print("this config avg is: {}".format(statistics.mean(config.final_tokens_arr[p])))
            print("this config min value is: {}, max value is: {}".format(min(config.final_tokens_arr[p]), max(config.final_tokens_arr[p])))
            print("this is the total tokens: {}".format(sum(column(config.final_tokens_arr, 0))))
            print("length of final_tokens_arr[p] is: {}".format(len(config.final_tokens_arr[p])))
            
    
    if Debug is True:
        print(avg)
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, STD, width, label='STD')
    rects2 = ax.bar(x + width/2, avg, width, label='Avg')
    
    ax.set_ylabel('STD & Avg')
    ax.set_xlabel('Number of Rounds (per experiment)')
    ax.set_title('Avg & STD values as function of rounds_num')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()


# Graph8 - prints the average STD of the participants FOR MORE THAN ONE EPOCH as function of the exp_num of the simulation
# Note: can work only if there's more than one configuration 
def Graph8(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    used_exp_nums = []
    
    # define the epochs you want to plot
    epoch1 = 1
    epoch2 = 5
    epoch3 = 10
    
    epoch1_avg_STDs = []
    epoch2_avg_STDs = []
    epoch3_avg_STDs = []
    
    # sort the Configs_list by the key of exp_num
    Configs_list.sort(key=Exp_num)
    
    if (Debug is True):
        print("config num is: {}".format(len(Config_list)))
        
    for config in Config_list:
        tmp_exp_num = config.exp_num
        if (Debug is True):
            print("tmp_exp_num is: {}".format(tmp_exp_num))
        
        if tmp_exp_num not in used_exp_nums:
            labels.append(str(tmp_exp_num))
            used_exp_nums.append(tmp_exp_num)
        
        # creating a list of all STDs of this config
        for j in range(config.p_num):
            STD.append(statistics.stdev(config.final_tokens_arr[j]))
        if (config.epoch == epoch1):
            epoch1_avg_STDs.append(mean(STD))
        elif (config.epoch == epoch2):
            epoch2_avg_STDs.append(mean(STD))
        elif (config.epoch == epoch3):
            epoch3_avg_STDs.append(mean(STD))
        STD.clear()
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    label_name1 = 'Avg STD - epoch={}'.format(epoch1)
    label_name2 = 'Avg STD - epoch={}'.format(epoch2)
    label_name3 = 'Avg STD - epoch={}'.format(epoch3)
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, epoch1_avg_STDs, width, label=label_name1)
    rects2 = ax.bar(x - width/2, epoch2_avg_STDs, width, label=label_name2)
    rects3 = ax.bar(x + width/2, epoch3_avg_STDs, width, label=label_name3)
    
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Experiments')
    ax.set_title('Avg STD values as function of exp_num for different epoch sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.rand_seed,config.rounds_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.41, -0.05, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    

# Graph9 - prints the average STD of the participants FOR MORE THAN ONE EPOCH as function of the rounds_num of the simulation
# Note: can work only if there's more than one configuration 
def Graph9(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate Graph9 - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    used_rounds_nums = []
    
    # define the epochs you want to plot
    epoch1 = 1
    epoch2 = 10
    epoch3 = 50
    
    epoch1_avg_STDs = []
    epoch2_avg_STDs = []
    epoch3_avg_STDs = []
    
    # sort the Configs_list by the key of rounds_num
    Configs_list.sort(key=Rounds_num)
    
    if (Debug is True):
        print("config num is: {}".format(len(Config_list)))
        
    for config in Config_list:
        tmp_rounds_num = config.rounds_num
        if (Debug is True):
            print("tmp_rounds_num is: {}".format(tmp_rounds_num))
        
        if (tmp_rounds_num not in used_rounds_nums):
            labels.append(str(tmp_rounds_num))
            used_rounds_nums.append(tmp_rounds_num)
        
        # creating a list of all STDs of this config
        for j in range(config.exp_num):
            STD.append(statistics.stdev(column(config.final_tokens_arr, j)))
            
        total_tokens = sum(column(config.final_tokens_arr, 0))    
        if (config.epoch == epoch1):
            epoch1_avg_STDs.append(mean(STD)/total_tokens)
        elif (config.epoch == epoch2):
            epoch2_avg_STDs.append(mean(STD)/total_tokens)
        elif (config.epoch == epoch3):
            epoch3_avg_STDs.append(mean(STD)/total_tokens)
        STD.clear()
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    label_name1 = 'Avg STD - epoch={}'.format(epoch1)
    label_name2 = 'Avg STD - epoch={}'.format(epoch2)
    label_name3 = 'Avg STD - epoch={}'.format(epoch3)
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, epoch1_avg_STDs, width, label=label_name1)
    rects2 = ax.bar(x - width/2, epoch2_avg_STDs, width, label=label_name2)
    rects3 = ax.bar(x + width/2, epoch3_avg_STDs, width, label=label_name3)
    
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Rounds per Experiments')
    ax.set_title('Avg STD values as function of rounds_num for different epoch sizes (Normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of experiment: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.rand_seed,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.37, -0.09, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()


# Graph10 - prints the average STD & one participant's RELATIVE tokens at the begining and at the end of the experiments as function of start_coins
# Note: can work only if there's more than one configuration 
def Graph10(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate Graph10 - Not enough CSV files.")
        sys.exit()
    
    labels = []
    STD = []
    
    start_means = []
    end_means = []
    avg_STD = []
    
    used_start_coins = []
    rand_p = 1
    
    # sort the Configs_list by the key of start_coins
    Configs_list.sort(key=Start_Coins)
    
    if (Debug is True):
        print("config num is: {}".format(len(Config_list)))
        
    for config in Config_list:
    
        total_start_tokens = config.p_num*config.start_coins
        total_end_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size)
        start_rel_weight = config.start_coins/total_start_tokens
        start_means.append(start_rel_weight)
        end_means.append((sum(config.final_tokens_arr[rand_p]))/(config.exp_num*total_end_tokens))
        
        tmp_start_coins = config.start_coins
        
        if (Debug is True):
            print("tmp_start_coins is: {}".format(tmp_start_coins))
        
        if (tmp_start_coins not in used_start_coins):
            labels.append(str(tmp_start_coins))
            used_start_coins.append(tmp_start_coins)
        
        # creating a list of all STDs of this config
        for j in range(config.exp_num):
            STD.append(statistics.stdev(column(config.final_tokens_arr, j)))
        
        total_tokens = sum(column(config.final_tokens_arr, 0))    
        avg_STD.append(mean(STD)/total_tokens)
        STD.clear()
    
    
    x = np.arange(len(labels))  # the label locations
    width = 0.20  # the width of the bars

    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, start_means, width, label='Relative weight - Start')
    rects2 = ax.bar(x - width/2, end_means, width, label='Mean relative weight - End')
    rects3 = ax.bar(x + width/2, avg_STD, width, label='Avg STD value')
    
    ax.set_xlabel('Number of Start coins for each participant')
    ax.set_title('Avg STD values, Relative weights - Start & End as function of start_coins')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower left")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiment: {}\nEpoch size: {}\nNumber of coins per winning: {} ".format(config.p_num,config.rand_seed,config.rounds_num,config.exp_num,config.epoch,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.41, -0.09, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    

# Graph11 - prints one participant's RELATIVE tokens at the begining and at the end of the experiments, for several configurations (mean value)
def Graph11(Configs_list, figure_num):
    # checking if there is more than one config
    if CSV_files_num <= 1:
        print("Can't generate MaxMinAvg STD graph - Not enough CSV files.")
        sys.exit()
    
    start_means = []
    end_means = []

    labels = []
    
    rand_p = 0;
    
    for config in Config_list:
        total_start_tokens = config.p_num*config.start_coins
        total_end_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size)
        start_rel_weight = config.start_coins/total_start_tokens
        labels.append(config.rounds_num)
        start_means.append(start_rel_weight)
        end_means.append((sum(config.final_tokens_arr[rand_p]))/(config.exp_num*total_end_tokens))
 
    x = np.arange(len(labels))  # the label locations
    
    if(Debug is True):
        print(x)
    
    width = 0.20  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, start_means, width, label='Relative weight - Start')
    rects2 = ax.bar(x + width/2, end_means, width, label='Mean relative weight - End')

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Relative weights')
    ax.set_title('Relative weights - Start vs End')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    
    textstr = "Number of participants: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins per winning: {} ".format(config.p_num,config.rand_seed,config.rounds_num,config.exp_num,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(0.41, -0.05, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
# We will create a list of Configurations types.
# A loop will go over all the CSV files (list of strings holding the files' names) and create the Configurations and add them to the list. 
CSV_files_list = []
Config_list = []

# choose a folder with CSV files according to the graph you want to produce
CSV_files_list = glob.glob("graph1/*.csv")                   
CSV_files_num = len(CSV_files_list)

if Debug is True:
    print("CSV_files_num is: {}".format(CSV_files_num))
 
for file_name in CSV_files_list:
    p_ = 0
    is_config_created = 0 #unlock
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0]=="Parameters:":
                continue
            elif row[0]=="Participants number":
                p_num = int(row[1])
                continue
            elif row[0]=="Rounds before dividend":
                epoch = int(row[1])
                continue
            elif row[0]=="Rounds per experiment":
                rounds_num = int(row[1])
                continue
            elif row[0]=="Experiments number":
                exp_num = int(row[1])
                continue
            elif row[0]=="Winning size":
                win_size = int(row[1])
                continue
            elif row[0]=="Start coins number":
                start_coins = int(row[1])
                continue
            elif row[0]=="Random seed":
                rand_seed = int(row[1])
                continue
            elif row[0]=="Final tokens for each participant at each experiment:":
                continue
            # after parsing all the parameters - initiate the configuration class instance
            if is_config_created==0:
                is_config_created = 1 #lock
                curr_config = Configuration(
                    p_num, # p_num
                    epoch, # epoch
                    rounds_num, # rounds_num
                    exp_num, # exp_num
                    win_size, # win_size
                    start_coins, # start_coins
                    rand_seed)  # rand_seed
            
            if row[0][0]=="p":
                tmp_str = row[1][1:len(row[1])-1] 
                curr_config.FinalTArray_init(p_, list(tmp_str.split(", ")))
                p_ += 1
               
    Config_list.append(curr_config) 


# ******************** start of graph production ********************

participant_num = 5 # random choice...

### Choose a graph you want to plot ###

Graph1(Config_list, 1)
# Graph2(Config_list[0], 1)
# Graph3(Config_list[0], 2)
# Graph4(Config_list[0], 1)
# Graph5(Config_list, 1)
# Graph6(Config_list, 1)
# Graph7(Config_list, participant_num, 5)
# Graph8(Config_list, 1)
# Graph9(Config_list, 1)
# Graph10(Config_list, 1)
# Graph11(Config_list, 1)


plt.show()





