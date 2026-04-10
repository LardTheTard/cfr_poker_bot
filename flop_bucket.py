from itertools import combinations
import random
from collections import Counter
from pokerkit import State, StandardHighHand

def 


# ── Equity (EHS) ──────────────────────────────────────────────────────────────

def compute_ehs(
    state: State,
    n_samples: int = 1000
) -> float:
    """
    Equity vs random opponent hand, Monte Carlo rollout to river.
    Works for flop (3 board cards), turn (4), or river (5).
    """
    p0_holes = ''.join(repr(c) for c in state.hole_cards[0] if c is not None)
    p1_holes = ''.join(repr(c) for c in state.hole_cards[1] if c is not None)
    board = ''
    for card in state.board_cards: 
        board += repr(card[0])
    p0 = StandardHighHand.from_game(p0_holes, board)
    p1 = StandardHighHand.from_game(p1_holes, board)

    known = hole + board
    deck = state.deck_cards.copy()
    cards_to_deal = 5 - int(len(board) / 2) # how many board cards remain
    wins = ties = 0

    for _ in range(n_samples):
        sample = random.sample(deck, 2 + cards_to_deal)
        opp_hole = sample[:2]
        runout = board + sample[2:] # full 5-card board

        my_score = evaluate_7(hole + runout)
        opp_score = evaluate_7(opp_hole + runout)

        if my_score > opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / n_samples


# ── Potential (PPOT / NPOT) ────────────────────────────────────────────────────

def compute_potential(
hole: list[int],
board: list[int],
n_samples: int = 1000
) -> tuple[float, float]:
    """
    PPOT: P(behind now, ahead at river)
    NPOT: P(ahead now, behind at river)

    Only meaningful on flop and turn (not river — no cards left).
    For flop: samples one turn card then one river card.
    For turn: samples one river card.
    """
    assert len(board) in (3, 4), "Potential only for flop/turn"
    known = hole + board
    deck = remaining_deck(known)
    cards_to_deal = 5 - len(board)

    ahead_now_behind_later = 0
    behind_now_ahead_later = 0
    ahead_now_total = 0
    behind_now_total = 0

    for _ in range(n_samples):
        sample = random.sample(deck, 2 + cards_to_deal)
        opp_hole = sample[:2]
        runout = board + sample[2:]

        my_now = evaluate_7(hole + board) # current street equity proxy
        opp_now = evaluate_7(opp_hole + board)

        my_final = evaluate_7(hole + runout)
        opp_final = evaluate_7(opp_hole + runout)

        currently_ahead = my_now > opp_now
        currently_behind = my_now < opp_now

        if currently_ahead:
            ahead_now_total += 1
            if my_final < opp_final:
                ahead_now_behind_later += 1

        if currently_behind:
            behind_now_total += 1
            if my_final > opp_final:
                behind_now_ahead_later += 1

    ppot = behind_now_ahead_later / behind_now_total if behind_now_total > 0 else 0.0
    npot = ahead_now_behind_later / ahead_now_total if ahead_now_total > 0 else 0.0
    return ppot, npot


def compute_ehs2(hole, board, n_samples=1000) -> float:
    """Combined EHS + potential metric. Better single feature than EHS alone."""
    ehs = compute_ehs(hole, board, n_samples)
    ppot, npot = compute_potential(hole, board, n_samples)
    return ehs + (1 - ehs) * ppot - ehs * npot


# ── Board texture ──────────────────────────────────────────────────────────────

def board_flush_texture(board: list[int]) -> str:
    """
    Returns 'monotone', 'two_tone', or 'rainbow'.
    Valid for flop (3 cards) and turn (4 cards).
    On river this is less meaningful — use draw_completed instead.
    """
    suit_counts = Counter(suit(c) for c in board)
    max_suit = max(suit_counts.values())
    if max_suit >= 3:
        return 'monotone'
    if max_suit == 2:
        return 'two_tone'
    return 'rainbow'


def board_is_paired(board: list[int]) -> bool:
    """True if any rank appears more than once on the board."""
    ranks = [rank(c) for c in board]
    return len(ranks) != len(set(ranks))


# ── Draw completed (turn and river) ───────────────────────────────────────────

def flush_draw_completed(prev_board: list[int], new_board: list[int]) -> bool:
    """
    Did a flush become possible (or complete) with the new card?
    prev_board = board before this street's card
    new_board = board including this street's card
    """
    def max_suited(b):
        if not b:
            return 0
        return max(Counter(suit(c) for c in b).values())
    return max_suited(new_board) >= 3 and max_suited(prev_board) < 3


def straight_draw_completed(prev_board: list[int], new_board: list[int]) -> bool:
    """
    Did a straight become possible (3+ connected ranks) with the new card?
    Uses a sliding window of 5 consecutive ranks.
    """
    def max_straight_draw(b):
        if not b:
            return 0
        rs = set(rank(c) for c in b)
        best = 0
        for low in range(9): # 2-through-T as low card of a straight
            window = sum(1 for r in range(low, low + 5) if r in rs)
            best = max(best, window)
        # Check wheel (A-2-3-4-5): ace counts as rank 12, treat as -1
        if 12 in rs:
            wheel = sum(1 for r in [-1,0,1,2,3] if (r == -1 or r in rs))
            best = max(best, wheel)
        return best
    return max_straight_draw(new_board) >= 3 and max_straight_draw(prev_board) < 3


# ── Bucketing functions ────────────────────────────────────────────────────────
# These convert raw floats to discrete bucket indices.

def equity_bucket(ehs: float, n_bins: int = 8) -> int:
    """0 to n_bins-1"""
    return min(int(ehs * n_bins), n_bins - 1)

def potential_bucket(ppot: float, n_bins: int = 4) -> int:
    return min(int(ppot * n_bins), n_bins - 1)


# ── Street-level bucket builders ──────────────────────────────────────────────

def flop_card_bucket(
hole: list[int],
board: list[int], # 3 cards
n_samples: int = 500
) -> dict:
    assert len(board) == 3
    ehs = compute_ehs(hole, board, n_samples)
    ppot, npot = compute_potential(hole, board, n_samples)
    return {
    'ehs': ehs,
    'ppot': ppot,
    'npot': npot,
    'ehs_bucket': equity_bucket(ehs),
    'ppot_bucket': potential_bucket(ppot),
    'npot_bucket': potential_bucket(npot),
    'flush_texture': board_flush_texture(board),
    'board_paired': board_is_paired(board),
    }


def turn_card_bucket(
hole: list[int],
board: list[int], # 4 cards
flop: list[int], # first 3 cards (to detect draw completion)
n_samples: int = 500
) -> dict:
    assert len(board) == 4
    ehs = compute_ehs(hole, board, n_samples)
    ppot, npot = compute_potential(hole, board, n_samples)
    return {
    'ehs': ehs,
    'ppot': ppot,
    'npot': npot,
    'ehs_bucket': equity_bucket(ehs),
    'ppot_bucket': potential_bucket(ppot),
    'npot_bucket': potential_bucket(npot),
    'flush_texture': board_flush_texture(board),
    'board_paired': board_is_paired(board),
    'flush_draw_completed': flush_draw_completed(flop, board),
    'straight_draw_completed': straight_draw_completed(flop, board),
    }


def river_card_bucket(
hole: list[int],
board: list[int], # 5 cards
turn_board: list[int], # first 4 cards (to detect draw completion)
n_samples: int = 500
) -> dict:
    assert len(board) == 5
    ehs = compute_ehs(hole, board, n_samples) # pure equity, no potential
    return {
    'ehs': ehs,
    'ehs_bucket': equity_bucket(ehs, n_bins=8),
    'flush_draw_completed': flush_draw_completed(turn_board, board),
    'straight_draw_completed': straight_draw_completed(turn_board, board),
    'board_paired': board_is_paired(board),
    }


# ── Combined infoset key ───────────────────────────────────────────────────────

def make_card_infoset_key(street: int, bucket_dict: dict) -> tuple:
    """
    Produces a hashable tuple suitable as the card component of an infoset key.
    Combine this with your history bucket for the full key.
    """
    if street == 1: # flop
        return (
        bucket_dict['ehs_bucket'],
        bucket_dict['ppot_bucket'],
        bucket_dict['flush_texture'],
        bucket_dict['board_paired'],
        )
    elif street == 2: # turn
        return (
        bucket_dict['ehs_bucket'],
        bucket_dict['ppot_bucket'],
        bucket_dict['flush_texture'],
        bucket_dict['board_paired'],
        bucket_dict['flush_draw_completed'],
        bucket_dict['straight_draw_completed'],
        )
    elif street == 3: # river
        return (
        bucket_dict['ehs_bucket'],
        bucket_dict['flush_draw_completed'],
        bucket_dict['straight_draw_completed'],
        bucket_dict['board_paired'],
        )