import random

# Kuhn poker uses 3 cards: J, Q, K
CARDS = [1, 2, 3]  # 1=J, 2=Q, 3=K


class Node:
    def __init__(self, info_set):
        self.info_set = info_set
        self.num_actions = 2  # pass/check or bet/call
        self.regret_sum = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def get_strategy(self, realization_weight):
        strategy = [0.0, 0.0]
        normalizing_sum = 0.0

        # Regret matching
        for a in range(self.num_actions):
            strategy[a] = max(self.regret_sum[a], 0.0)
            normalizing_sum += strategy[a]

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / self.num_actions

            self.strategy_sum[a] += realization_weight * strategy[a]

        return strategy

    def get_average_strategy(self):
        avg_strategy = [0.0, 0.0]
        normalizing_sum = sum(self.strategy_sum)

        for a in range(self.num_actions):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self.num_actions

        return avg_strategy


class KuhnCFR:
    def __init__(self):
        self.node_map = {}

    def is_terminal(self, history):
        # Terminal histories in Kuhn poker
        return history in ["pp", "bb", "bp", "pbp", "pbb"]

    def terminal_utility(self, cards, history):
        # Utility is from player 0's perspective
        player0_card = cards[0]
        player1_card = cards[1]

        # Showdown after both pass
        if history == "pp":
            return 1 if player0_card > player1_card else -1

        # Someone bet and other folded
        if history == "bp":
            return 1
        if history == "pbp":
            return -1

        # Bet called, pot is bigger
        if history == "bb" or history == "pbb":
            return 2 if player0_card > player1_card else -2

        raise ValueError(f"Unknown terminal history: {history}")

    def cfr(self, cards, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if self.is_terminal(history):
            util = self.terminal_utility(cards, history)
            return util if player == 0 else -util

        info_set = str(cards[player]) + history

        if info_set not in self.node_map:
            self.node_map[info_set] = Node(info_set)
        node = self.node_map[info_set]

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = [0.0, 0.0]
        node_util = 0.0

        # Action 0 = pass/check/fold
        # Action 1 = bet/call
        for a in range(2):
            next_history = history + ("p" if a == 0 else "b")

            if player == 0:
                util[a] = -self.cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self.cfr(cards, next_history, p0, p1 * strategy[a])

            node_util += strategy[a] * util[a]

        # Regret update
        for a in range(2):
            regret = util[a] - node_util
            if player == 0:
                node.regret_sum[a] += p1 * regret
            else:
                node.regret_sum[a] += p0 * regret

        return node_util

    def train(self, iterations):
        util = 0.0

        for _ in range(iterations):
            cards = CARDS[:]
            random.shuffle(cards)
            util += self.cfr(cards, "", 1.0, 1.0)

        print(f"Average game value for player 1: {util / iterations:.6f}")
        print()

        for info_set in sorted(self.node_map):
            avg = self.node_map[info_set].get_average_strategy()
            print(
                f"{info_set}: pass/check={avg[0]:.3f}, bet/call={avg[1]:.3f}"
            )


if __name__ == "__main__":
    trainer = KuhnCFR()
    trainer.train(100000)