import os
import h5py
import ast
from collections import deque
import numpy as np
from collections import Counter
from numpy import savez_compressed


game_folders = ['MO/', 'PLAY/', 'LIU/']
folder_name = 'output2017/'
all_game_files = []

for game_folder in game_folders:
    for file_name in os.listdir(folder_name + game_folder):
        all_game_files.append(folder_name + game_folder + file_name)
        
        
w_dict = {'W' + str(i+1): i for i in range(9)} #万
b_dict = {'B' + str(i+1): i+9 for i in range(9)} #饼
t_dict = {'T' + str(i+1): i+18 for i in range(9)} #条
f_dict = {'F' + str(i+1): i+27 for i in range(4)} #风 东南西北
j_dict = {'J' + str(i+1): i+31 for i in range(3)} #（剑牌）中发白
total_dict = {**w_dict, **b_dict,**t_dict,**f_dict,**j_dict}

def serialize(array):
    res = np.zeros((4,34),int)
    count = Counter(array)
    for i in count:
        if i[0] != 'H':
            index = total_dict[i]
            nums = count[i]
            for j in range(nums):
                res[j][index] = 1
    return np.expand_dims(np.array(res), axis = 0)

def serialize_y(tile):
    res=np.zeros((34),int)
    index = total_dict[tile]
    res[index] = 1
    return res

def update_history(past_game_history, player_tile, discard_tile_list, stealing_tile_list, round_tile):
    X = serialize(player_tile) 
    
    # Only encode the current card for the past history games
    X = np.concatenate((X, serialize([round_tile])), axis = 0)#历史手牌拼接到当前手牌上
    
    # Encoding all the discard tile
    for discard_list in discard_tile_list:
        X = np.concatenate((X, serialize(discard_list)), axis = 0)#历史出牌拼接到当前手牌上
        
    # Encoding all the stealing tile
    for stealing_list in stealing_tile_list:
        X = np.concatenate((X, serialize(stealing_list)), axis = 0)#历史摸牌拼接到当前手牌上

    past_game_history.append(X) # Append the current situation into the history queue
    past_game_history.popleft() # Remove the last information history
    
    return past_game_history


def extract_feature_X(player_tile, discard_tile, stealing_tile, past_game_history, player_num):
    """Feature encoding""" 
    # Encoding own hand feature
    X = serialize(player_tile) 

    # Encoding all the discard tile
    for discard_list in discard_tile:
        X = np.concatenate((X, serialize(discard_list)), axis = 0)
        
    # Encoding all the stealing tile
    for stealing_list in stealing_tile:
        X = np.concatenate((X, serialize(stealing_list)), axis = 0) 

    # Encode the past history situation into feature
    for past_game_list in past_game_history:
        X = np.concatenate((X, past_game_list), axis = 0) 
        
    if player_num == 0:
        wind_feature = np.expand_dims(np.ones((4,34),int), axis = 0)
        X = np.concatenate((X,wind_feature), axis = 0) 
    else:
        wind_feature = np.expand_dims(np.zeros((4,34),int), axis = 0)
        X = np.concatenate((X,wind_feature), axis = 0) 
   
    return X


def generate_training_set(file_name, player_num, history_num = 4):
    """
    Player tile : 1
    discard tile : 4
    stealing tile : 4
    last 1 game history : 4+4+1+1 = 10(last discard)
    last 2 game history : 4+4+1+1 = 10(last discard)
    last 3 game history : 4+4+1+1 = 10(last discard)
    """
    
    other_player = [i for i in range(4)]
    other_player.pop(player_num)

    # Initialize the empty array for data storage
    master_X = []
    master_Y = []
    discard_tile_list = [[],[],[],[]]
    stealing_tile_list = [[],[],[],[]]
    past_game_history = deque([np.zeros((10, 4, 34),int) for x in range(history_num)])

    # Extract the whole dataset
    with open(file_name, 'r', encoding='utf8') as f:
        all_data_str = f.readlines()

    # Only getting the player own tiles
    player_hand = all_data_str[2 + player_num]
    start = player_hand.find('[')
    end = player_hand.find(']') + 1
    player_tile = ast.literal_eval(player_hand[start:end])

    for round_info in all_data_str[6:]:
        round_info = round_info.split('\t')

        # Extracting basic information
        round_player_num = int(round_info[0])
        action = round_info[1]
        round_tile = ast.literal_eval(round_info[2])[0]

        if len(round_info) == 5:
            round_tile = ast.literal_eval(round_info[2])
            stealing_tile = round_info[3]

        # First stored the current infomation into data record
        if action == '打牌':
            X = extract_feature_X(player_tile, discard_tile_list, stealing_tile_list, past_game_history, player_num)
            Y = serialize_y(round_tile)
            master_X.append(X)
            master_Y.append(Y)

        # Then proceed to game play logic
        if action == '打牌':
            discard_tile_list[round_player_num].append(round_tile)
            if player_num == round_player_num:
                player_tile.remove(round_tile)

        elif action in ('摸牌', '补花后摸牌', '杠后摸牌'):
            if player_num == round_player_num:
                player_tile.append(round_tile)
            else:
                pass

        elif action in ('吃', '碰', '明杠'):
            # Adding the information into the stealing tile
            for tile in round_tile:
                stealing_tile_list[round_player_num].append(tile) 

            if player_num == round_player_num:
                # Remove the tile from player own hand
                round_tile.remove(stealing_tile)
                for tile in round_tile:
                    player_tile.remove(tile)
            else:
                pass

        # Finally update the history record 
        if action == '打牌':
            past_game_history = update_history(past_game_history, player_tile, discard_tile_list, stealing_tile_list, round_tile)
            
    return master_X, master_Y



if __name__ == '__main__':
    master_X = []
    master_Y = []

    old_file_count = 0
    saving_file_count = 0
    file_count = 0

    for file_name in all_game_files:
        for player_num in range(0,4):
            X, Y = generate_training_set(file_name, player_num, history_num = 4)
            # print('initial x',X[0][0])
            # print('initial y',Y[0])
            # print('shape of X', np.array(X).shape)
            master_X += X
            master_Y += Y
            # print(np.array(master_X).shape)
            # print('mx',master_X[0][0:2])
            # print('my',master_Y[0:2])

        if len(master_X) > 100000:
            import bloscpack as bp
            master_X = np.array(master_X)
            master_Y = np.array(master_Y)
            saving_file_count += len(master_X)
            bp.pack_ndarray_to_file(master_X, f'processed_data_blp/input_X_{file_count}_{saving_file_count}.nosync.blp')
            bp.pack_ndarray_to_file(master_Y, f'processed_data_blp/input_Y_{file_count}_{saving_file_count}.nosync.blp')
            file_count += len(master_X)
            print('Saved with the file size', saving_file_count)
            
            master_X = []
            master_Y = []
            break
            
