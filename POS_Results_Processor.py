import random
### import argparse
from collections import namedtuple
import sys
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import statistics
from statistics import mean
from matplotlib.offsetbox import AnchoredText

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
        

# We will create a list of Configurations types.
# A loop will go over all the CSV files (list of strings holding the files' names) and create the Configurations and add them to the list. 
CSV_files_list = ['00010_00001_00200_00100_00001_00001_00034','00010_00001_00150_00100_00001_00001_00034','00010_00001_00300_00100_00001_00001_00034',
                    '00010_00001_00250_00100_00001_00001_00034'] 
CSV_files_num = len(CSV_files_list)

Config_list = []
      
for file_name in CSV_files_list:
    p_ = 0
    is_config_created = 0 #unlock
    with open(file_name+'.csv') as csvfile:
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
    
def Exp_num(elem):
    return elem.exp_num

def Rounds_num(elem):
    return elem.rounds_num
    
# prints a given participant tokens histogram (final tokens for each experiment)
def PlotParticipantTokensHist(config, p, figure_num):
    exp_arr = []
    for i in range(config.exp_num):
        exp_arr.append(i)
    plt.figure(figure_num)
    ax = plt.subplot()
    ax.plot(exp_arr, config.final_tokens_arr[p], 'g')
    plt.xlabel('Experiment')
    plt.ylabel('Final tokens')
    plt.title('Final tokens for each experiment for participant {}\n\n mean value is {}'.format(p, mean(config.final_tokens_arr[p])))
    plt.grid(True)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(p_num,epoch,rand_seed,rounds_num,exp_num,start_coins,win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props)
    
# prints a given participant RELATIVE tokens ((final tokens)/(total tokens) for each experiment )
def PlotParticipantRelativeTokens(config, p, figure_num):
    exp_arr = []
    rel_weight = []
    total_tokens = (config.p_num*config.start_coins) + (config.rounds_num*config.win_size)
    for i in range(config.exp_num):
        exp_arr.append(i)
        rel_weight.append((config.final_tokens_arr[p][i])/total_tokens)
    plt.figure(figure_num)
    ax = plt.subplot()
    ax.plot(exp_arr, rel_weight, 'b')
    #plt.hist(rel_weight, bins=exp_num)
    plt.xlabel('Experiment')
    plt.ylabel('$\dfrac{Final\ tokens}{Total\ tokens}$')
    plt.title('$\dfrac{Final\ tokens}{Total\ tokens}$ ' + 'for each experiment for participant {}\n\n mean value is {}\
    \n\n Total tokens: {}'.format(p, "%.3f" % mean(rel_weight), total_tokens))
    plt.grid(True)

    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props)
    
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format("%.3f" % height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# prints each participant's RELATIVE tokens at the begining and at the end of the experiments (mean value)
def PlotParticipatsStartEndRelativeTokensHist(config, figure_num):
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
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    
# prints the standart deviation of each participant of the final number of tokens for each experiment
def PlotSTD_P(config, figure_num):
    STD = []
    labels = []
    
    for j in range(config.p_num):
        labels.append("p"+str(j+1))
        STD.append(statistics.stdev(config.final_tokens_arr[j]))
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, STD, width, label='Standard Deviation - each participant')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation of each participant')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    
# prints the standart deviation of all the participants for each experiment
def PlotSTD_ALL(config, figure_num):
    STD_all = []
    labels = []
    temp_arr = [0]*config.p_num   
    for i in range (config.exp_num):
        labels.append("exp"+str(i+1))
        for j in range(config.p_num):
            temp_arr[j]=config.final_tokens_arr[j][i]
        STD_all.append(statistics.stdev(temp_arr))
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, STD_all, width, label='Standard Deviation - each experiment')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Standard Deviation of each experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    
# prints the min, max and average STD of the participants as function of the exp_num of the simulation
# Note: can work only if there's more than one configuration 
def PlotMinMaxAvgSTD_ExpNum(Configs_list, figure_num):
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
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Experiments')
    ax.set_title('Min,Avg,Max STD values as function of exp_num')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    
# prints the min, max and average STD of the participants as function of the Rounds per experiment of the simulation
# Note: can work only if there's more than one configuration 
def PlotMinMaxAvgSTD_RoundNum(Configs_list, figure_num):
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
        if tmp_rounds_num in used_rounds_nums:
            continue
        used_rounds_nums.append(tmp_rounds_num)
        labels.append(str(tmp_rounds_num))
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
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD')
    ax.set_xlabel('Number of Rounds (per experiment)')
    ax.set_title('Min,Avg,Max STD values as function of rounds_num')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()

# prints the mean and STD of the a specific participant & exp_num as function of the Rounds per experiment of the simulation
# Note: can work only if there's more than one configuration 
def P_AvgSTD_RoundNum(Configs_list, p, figure_num):
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
            continue
        used_rounds_nums.append(tmp_rounds_num)
        labels.append(str(tmp_rounds_num))
        # creating a list of all STDs of this config
        STD.append(statistics.stdev(config.final_tokens_arr[p]))
        avg.append(statistics.mean(config.final_tokens_arr[p]))
    
    x = np.arange(len(labels))  # the label locations
    width = 0.10  # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, STD, width, label='STD')
    rects2 = ax.bar(x + width/2, avg, width, label='Avg')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('STD & Avg')
    ax.set_xlabel('Number of Rounds (per experiment)')
    ax.set_title('Avg & STD values as function of rounds_num')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc = "lower right")
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()


    
### PlotParticipantTokensHist(curr_config, 1, 1)
### PlotParticipantRelativeTokens(curr_config, 1, 2)
### PlotSTD_ALL(curr_config, 2)


PlotSTD_P(Config_list[0], 1)
PlotParticipatsStartEndRelativeTokensHist(Config_list[0], 2)
PlotMinMaxAvgSTD_ExpNum(Config_list, 3)
PlotMinMaxAvgSTD_RoundNum(Config_list, 4)
P_AvgSTD_RoundNum(Config_list,1, 5)
plt.show()





