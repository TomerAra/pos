import random
import argparse
from collections import namedtuple
import sys
import csv
from tqdm import tqdm
import numpy as np
import os.path


Participant = namedtuple("Participant", ["Name", 'Tokens'])
U_resolution = 1000000

################################### START ARGUMENTS ###################################

parser = argparse.ArgumentParser(description='Script purpose: POS simulation')
requiredNamed = parser.add_argument_group('Required Arguments')

# Seed
requiredNamed.add_argument('-seed',metavar='Seed', type=int,
                    help='Seed for random function. Example: -seed 30'  ,required=True)
                    
args = parser.parse_args()

if args.seed is None:
	sys.exit("Error: no seed argument")

rand_seed = args.seed

################################### END ARGUMENTS ###################################

################################### START CONFIGURATIONS ###################################

## list of lists:
configurations_list = []

## configuration template is:
## configuration = [p_num, epoch, rounds_num, exp_num, win_size, start_coins]

configurations_list.append([10, 1, 250, 100, 1, 1])
configurations_list.append([10, 1, 300, 100, 1, 1])
configurations_list.append([10, 1, 150, 100, 1, 1])
configurations_list.append([10, 1, 200, 100, 1, 1])
# configurations_list[1] = [10, 1, 100, 100, 1, 100]

number_of_configurations = len(configurations_list)

################################### END CONFIGURATIONS ###################################

# select a random number in the range (0,1)
def RandomUselect(U_resolution):
	rand_int = random.randint(0,U_resolution-1)
	return (float(rand_int/U_resolution))
        
    
class Configuration:
    
    def __init__(self, p_num, epoch, rounds_num, exp_num, win_size, start_coins):
        self.p_num = p_num
        self.epoch = epoch
        self.rounds_num = rounds_num
        self.exp_num = exp_num
        self.win_size = win_size
        self.start_coins = start_coins
        self.P_arr = []
        self.P_weights_arr = []
        self.Dividends_arr = []
        self.final_tokens_arr = []
        self.seeds_list = []
        self.TotalTokens = 0
    
    # initializing P_arr with Tokens=start_coins and Dividends_arr and P_weights_arr with zeros
    def InitSimulation(self):
        self.seeds_list = random.sample(range(U_resolution), self.exp_num+1) #the range (0,U_resolution) was selected randomly...
        for i in range(self.p_num):
            temp_name = "p"+str(i+1)
            self.P_arr.append(Participant(Name=temp_name, Tokens=self.start_coins))
            self.Dividends_arr.append(0)
            self.P_weights_arr.append(0)
            tmp_list = []
            self.final_tokens_arr.append(tmp_list)
            
    # giving all the participants the tokens they won and zero the Dividends_arr	
    def DividendsGiving(self):
        for i in range(self.p_num):
            past_tokens = self.P_arr[i].Tokens
            self.P_arr[i] = self.P_arr[i]._replace(Tokens = past_tokens+self.Dividends_arr[i])
            self.Dividends_arr[i] = 0
            
    # calculating the sum off all tokens
    def CalcTotalWeight(self):
        self.TotalTokens = 0
        for i in range(self.p_num):
            self.TotalTokens += self.P_arr[i].Tokens
        return self.TotalTokens
    
    # calculating the weight of each participant
    def CalcPWeight(self):
        for i in range(self.p_num):
            self.P_weights_arr[i] = (float(self.P_arr[i].Tokens/self.TotalTokens))
    
    # calculating the winner for this round
    def CalcRoundResults(self, U):
        for i in range(self.p_num):
            if i==0:
                low = 0
            else:
                low = low + self.P_weights_arr[i-1]
            if U >= low and U < low+self.P_weights_arr[i]:
                return i
    
    # printing all participants name and tokens
    def PrintParticipants(self):
        for i in range(self.p_num):
            print("name: {}, Tokens: {}".format(self.P_arr[i].Name, self.P_arr[i].Tokens))
        print("\n")
    
    # printing final tokens array
    def PrintTokensArray(self):
        for i in range(self.p_num):
            print("name: {}, Final tokens in each experiment: {}".format(self.P_arr[i].Name, self.final_tokens_arr[i]))
            
    # updating the final tokens array (sums-up every experiment)
    def UpdateFinalTokensArray(self):
        for i in range(self.p_num):
            self.final_tokens_arr[i].append(self.P_arr[i].Tokens)
    
    # set players tokens to 1 (for the next experiment), Dividends_arr and P_weights_arr
    def ZeroArrays(self):
        for i in range(self.p_num):
            self.P_arr[i] = self.P_arr[i]._replace(Tokens = self.start_coins)
            self.Dividends_arr[i] = 0
            self.P_weights_arr[i] = 0
            
    # prints the configuration details
    def PrintConfigParameters(self, seed):
        print("\nParameters:")
        print("Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\n\
        Number of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(self.p_num,self.epoch,seed,self.rounds_num,self.exp_num,self.start_coins,self.win_size))
    
    # check if a simulation has aleardy ran with this configuration (if the matched CSV file exsits)
    def Check_if_exists(self):
        file_name = '{:05}_{:05}_{:05}_{:05}_{:05}_{:05}_{:05}.csv'.format(self.p_num,self.epoch,self.rounds_num,self.exp_num,self.win_size,self.start_coins,rand_seed)
        return os.path.isfile(file_name)
    
    
    # writes the results of the simulation with this configuration to a CSV file
    def ResultsToCSV(self):
        file_name = '{:05}_{:05}_{:05}_{:05}_{:05}_{:05}_{:05}.csv'.format(self.p_num,self.epoch,self.rounds_num,self.exp_num,self.win_size,self.start_coins,rand_seed)
        with open(file_name, 'w', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            results_writer.writerow(['Parameters:'])
            results_writer.writerow(['Participants number', self.p_num])
            results_writer.writerow(['Rounds before dividend', self.epoch])
            results_writer.writerow(['Rounds per experiment', self.rounds_num])
            results_writer.writerow(['Experiments number', self.exp_num])
            results_writer.writerow(['Winning size', self.win_size])
            results_writer.writerow(['Start coins number', self.start_coins])
            results_writer.writerow(['Random seed', rand_seed])
            
            results_writer.writerow(['Final tokens for each participant at each experiment:'])
            
            for i in range(self.p_num):
                results_writer.writerow(['p{}'.format(i+1),self.final_tokens_arr[i]])
           
            
            


            
random.seed(rand_seed)

print("\nStart:")

for c in tqdm(range(number_of_configurations)):
    # set the parameters
    curr_config = Configuration(
    configurations_list[c][0], # p_num
    configurations_list[c][1], # epoch
    configurations_list[c][2], # rounds_num
    configurations_list[c][3], # exp_num
    configurations_list[c][4], # win_size
    configurations_list[c][5])  # start_coins
    
    #print("\nParameters:")
    #print("Number of participants: {}\nRounds per dividend: {}\nSeed: {}\nNumber of rounds per experiment: {}\nNumber of experiments: {}\nNumber of coins initiated for each participant: {}\nNumber of coins per winning: {} ".format(curr_config.p_num,curr_config.epoch,rand_seed,curr_config.rounds_num,curr_config.exp_num,curr_config.start_coins,curr_config.win_size))
    
    # checking if this simulation already happened (if so, pass to the next configuration)
    if curr_config.Check_if_exists():
        del curr_config
        continue
    
    # initiation for this configuration
    curr_config.InitSimulation()
    Rounds_before_dividend_counter = 0
    
    # run simulation for this configuration 
    for j in tqdm(range(curr_config.exp_num)):
        for i in range(curr_config.rounds_num):
            tmp_U = RandomUselect(U_resolution)
            curr_config.TotalTokens = curr_config.CalcTotalWeight()
            curr_config.CalcPWeight()
            winner_index = curr_config.CalcRoundResults(tmp_U)
            curr_config.Dividends_arr[winner_index] += curr_config.win_size
            Rounds_before_dividend_counter += 1
            if Rounds_before_dividend_counter == curr_config.epoch:
                curr_config.DividendsGiving()
                Rounds_before_dividend_counter = 0
        
        curr_config.UpdateFinalTokensArray()
        new_seed = curr_config.seeds_list[j]
        random.seed(new_seed)
        curr_config.TotalTokens = 0
        curr_config.ZeroArrays()
        
    # write simulation results to a CSV file
    curr_config.ResultsToCSV()
    
    # reset for this configuration    
    del curr_config

print("\nFinish!")




