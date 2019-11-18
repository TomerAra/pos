import random
import argparse
from collections import namedtuple
import sys
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from statistics import mean
from matplotlib.offsetbox import AnchoredText

################################### START ARGUMENTS ###################################

parser = argparse.ArgumentParser(description='Script purpose: parsing results CSV file')
requiredNamed = parser.add_argument_group('Required Arguments')

# File name
requiredNamed.add_argument('-fname',metavar='File_name', type=str,
                    help='CSV file name. Example: -fname 10_1_100_10_1_1_3.csv'  ,required=True)
                    
args = parser.parse_args()

if args.fname is None:
	sys.exit("Error: no file name argument")

file_name = args.fname

################################### END ARGUMENTS ###################################

class Configuration:
    ### P_arr = []
    ### P_weights_arr = []
    ### Dividends_arr = []
    final_tokens_arr = []    
    
    def __init__(self, p_num, epoch, rounds_num, exp_num, win_size, start_coins, rand_seed):
        self.p_num = p_num
        self.epoch = epoch
        self.rounds_num = rounds_num
        self.exp_num = exp_num
        self.win_size = win_size
        self.start_coins = start_coins 
        self.rand_seed = rand_seed
        for i in range(self.p_num):
            tmp_list = []
            self.final_tokens_arr.append(tmp_list)

p_ = 0

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
            curr_config.final_tokens_arr[p_] = list(tmp_str.split(", "))
            for i in range(0, len(curr_config.final_tokens_arr[p_])): 
                curr_config.final_tokens_arr[p_][i] = int(curr_config.final_tokens_arr[p_][i])    
            p_ += 1
            
        
        
        
        
        
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
    
    ###text_box = AnchoredText(props, frameon=True, loc=2, pad=0.5)
    ###plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ###ax.add_artist(text_box)
    
    #plt.show()
	
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
        
    #plt.show()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format("%.4f" % height),
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
    ax.legend()
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    textstr = "Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(config.p_num,config.epoch,config.rand_seed,config.rounds_num,config.exp_num,config.start_coins,config.win_size)
    props = dict(boxstyle='round', facecolor='orange', alpha=0.5)
    
    # place a text box in upper left in axes coords
    plt.text(-0.30, 1.15, textstr, transform=ax.transAxes, fontsize=7,
        verticalalignment='top', bbox=props)
        
    fig.tight_layout()
    

    
    
#PlotParticipantTokensHist(curr_config, 1, 1)
#PlotParticipantRelativeTokens(curr_config, 1, 2)
PlotParticipatsStartEndRelativeTokensHist(curr_config, 3)
plt.show()





