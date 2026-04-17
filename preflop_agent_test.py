from exact_preflop_freqs import Node
from logger import Logger
from datetime import datetime
from play_hand import agent_vs_random, random_vs_random, full_agent_vs_random
from tqdm import tqdm
import pickle
import os
import shutil

PATH = r'C:\Users\login\RANDOM_CODE\wpt_bot\nodesets\nodes_2026-04-16_21-49-33.pkl'

def safe_name(s):
    return str(s).replace("/", "_").replace("\\", "_").replace(":", "_")

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs('logs', exist_ok=True)
os.makedirs('debug_logs', exist_ok=True)

logger = Logger(output_path=f"logs/{timestamp}.txt")
debug_logger = Logger(output_path=f"debug_logs/{timestamp}.txt")
game_logger = Logger(output_path="game_log.txt")
game_logger.clear_logs()

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

# optional: fully clear previous output
if os.path.exists('loaded_nodes'):
    shutil.rmtree('loaded_nodes')
os.makedirs('loaded_nodes', exist_ok=True)

# create one file per hand label
seen_hands = set()
for bucket, node in nodes.items():
    hand = safe_name(bucket[0])
    if hand not in seen_hands:
        with open(f'loaded_nodes/{hand}.txt', 'w', encoding='utf-8') as f:
            pass
        seen_hands.add(hand)

for bucket, node in nodes.items():
    hand = safe_name(bucket[0])
    with open(f'loaded_nodes/{hand}.txt', 'a', encoding='utf-8') as f:
        f.write('-----------------------------------------------------------------------------------\n')
        f.write(f"{bucket} - Loaded {node.times_visited} times\n")

        node_sum = sum(node.strategy_sum.values())
        if node_sum > 0:
            for k, v in node.strategy_sum.items():
                f.write(f"{k}: {round(v / node_sum * 100, 1)}%\n")
        else:
            f.write("No accumulated strategy yet (node_sum = 0)\n")

        for k, v in node.regret_sum.items():
            f.write(f"{k} regret: {round(v)}\n")

iters = 10_000

game_sum = 0
for _ in tqdm(range(iters)):
    reward = agent_vs_random(nodes, 1, game_logger)
    game_sum += reward

print("Average reward for agent v random:", game_sum / iters)