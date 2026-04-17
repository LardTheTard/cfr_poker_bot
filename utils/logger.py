from pokerkit import State

class Logger:
    def __init__(self, output_path) -> None:
        self.STREET_NAMES = {
            0: "Preflop",
            1: "Flop",
            2: "Turn",
            3: "River",
        }
        self.output_path = output_path
        pass

    def log(self, message: str) -> None:
        # print(message)
        with open(self.output_path, "a") as f:
            f.write(str(message) + "\n")

    def clear_logs(self) -> None:
        with open(self.output_path, "w") as f:
            f.write("")

    def log_street_state(self, state: State) -> None:
        street_name = self.STREET_NAMES.get(state.street_index, f"Street {state.street_index}")
        self.log(f"\n=== {street_name} ===")
        self.log(f"Board: {state.board_cards}")
        self.log(f"Total pot: {state.total_pot_amount}")
        self.log(f"Stacks: {state.stacks}\n")

    def log_final(self, state: State) -> None:
        self.log("\n=== FINAL ===")
        self.log(f"Hole cards: {state.hole_cards}")
        self.log(f"Final board: {state.board_cards}")
        self.log(f"Total pot: {state.total_pot_amount}")
        self.log(f"Final stacks: {state.stacks}")