from preflop_mccfr import Node
from logger import Logger
from datetime import datetime
from play_hand import agent_vs_random, random_vs_random
from tqdm import tqdm
import pickle
import os

# NAME = input("Input path to desired nodeset: ")
PATH = r'C:\Users\ZhaoLo\poker\cfr_poker_bot\nodesets\some_unexplored_200k.pkl'

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs('logs', exist_ok=True)
os.makedirs('debug_logs', exist_ok=True)

logger = Logger(output_path=f"logs/{timestamp}.txt")
debug_logger = Logger(output_path=f"debug_logs/{timestamp}.txt")
game_logger = Logger(output_path="game_log.txt")
game_logger.clear_logs() # clears logs so new hand can be logged

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

os.makedirs('loaded_nodes', exist_ok=True)

for bucket, node in nodes.items():
    with open(f'loaded_nodes/{bucket[0]}', 'w') as f:
        pass

for bucket, node in nodes.items():
    with open(f'loaded_nodes/{bucket[0]}', 'a') as f:
        f.write('-----------------------------------------------------------------------------------\n')
        f.write(f"{bucket} - Loaded {node.times_visited} times\n")
        node_sum = sum(node.strategy_sum.values())
        for k, v in node.strategy_sum.items():
            f.write(f"{k}: {round(v/node_sum * 100, 1)}\n")


iters = 100_000

game_sum = 0
for _ in tqdm(range(iters)):
    reward = agent_vs_random(nodes, 1, game_logger)
    game_sum += reward
print("Average reward for agent v random:", game_sum / iters)

game_sum = 0
for _ in tqdm(range(iters)):
    reward = random_vs_random(game_logger)
    game_sum += reward
print("Average reward for random v random:", game_sum / iters)