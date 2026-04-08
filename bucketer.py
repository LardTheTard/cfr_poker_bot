from pokerkit import State

class Bucketer:
    def __init__(self) -> None:
        pass

    def preflop_bucket(self, state: State, history: list) -> tuple:
        actor = state.actor_index
        hole_cards = str(state.hole_cards[actor]).strip('[]').split(', ')

        # ---------- card parsing ----------
        def rank_value(card):
            r = card[0]
            mapping = {
                "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12,
                "K": 13, "A": 14
            }
            return mapping[r]

        def suit(card):
            return card[1]

        r1 = rank_value(hole_cards[0])
        r2 = rank_value(hole_cards[1])
        s1 = suit(hole_cards[0])
        s2 = suit(hole_cards[1])

        high = max(r1, r2)
        low = min(r1, r2)
        suited = (s1 == s2)
        pair = (r1 == r2)
        gap = high - low

        # ---------- hand bucket ----------
        if pair:
            if high >= 11:  # JJ+
                hand_bucket = "premium_pairs"
            elif 8 <= high <= 10:  # 88-TT
                hand_bucket = "medium_pairs"
            else:  # 22-77
                hand_bucket = "small_pairs"

        elif suited and high == 14 and 10 <= low <= 13:
            # ATs, AJs, AQs, AKs
            hand_bucket = "premium_suited_aces"

        elif high == 14 and suited and 2 <= low <= 9:
            # A2s-A9s
            hand_bucket = "weak_suited_aces"

        elif high == 14 and not suited and low >= 12:
            # AKo, AQo
            hand_bucket = "premium_offsuit_aces"

        elif suited and high >= 11 and low >= 10:
            # KQs, KJs, QJs, JTs, AQs, AKs etc.
            # Since suited aces already handled above, this is mostly KQs-JTs type hands
            hand_bucket = "premium_suited_broadways"

        elif not suited and high >= 11 and low >= 10:
            # KQo / KJo / QJo / JTo / AQo / AJo / etc.
            if (high == 13 and low == 12):
                hand_bucket = "premium_offsuit_broadways"  # KQo
            elif (high == 14 and low >= 11):
                # AKo/AQo handled above, AJo falls here if you want it weaker
                hand_bucket = "weak_offsuit_broadways"
            elif high == 13 and low == 11:
                hand_bucket = "weak_offsuit_broadways"  # KJo
            elif high == 12 and low == 11:
                hand_bucket = "weak_offsuit_broadways"  # QJo
            elif high == 11 and low == 10:
                hand_bucket = "weak_offsuit_broadways"  # JTo
            else:
                hand_bucket = "trash_offsuit_hands"

        elif suited and gap == 1 and high <= 13:
            # suited connectors like 98s, T9s, 76s, KQs technically connector but already caught above
            hand_bucket = "suited_connectors"

        elif suited and gap == 2 and high <= 13:
            # suited one-gappers like 97s, T8s, J9s
            hand_bucket = "suited_gappers"

        else:
            hand_bucket = "trash_offsuit_hands"

        # ---------- position bucket ----------
        # Assumes heads-up: actor 0 = SB, actor 1 = BB
        position_bucket = "SB" if actor == 0 else "BB"

        # ---------- effective stack bucket ----------
        # Assumes state.stacks exists and blinds are normalized to bb units,
        # or at least consistent with 100bb+ interpretation.
        eff_stack = min(state.stacks[0], state.stacks[1])

        if eff_stack < 20:
            stack_bucket = "short"
        elif eff_stack < 50:
            stack_bucket = "medium"
        elif eff_stack < 100:
            stack_bucket = "deep"
        else:
            stack_bucket = "very_deep"

        # ---------- betting history bucket ----------
        if not history:
            history_bucket = "root"
        elif len(history) == 1:
            first = history[0]
            if first == 'check/call':
                history_bucket = "limped"
            elif first == 'raise':
                history_bucket = "open"
            else:
                history_bucket = "root"
        elif len(history) == 2:
            if (history[0] == 'raise') and (history[1] == 'raise'):
                history_bucket = "vs_3bet"
            else:
                history_bucket = "vs_open"
        elif len(history) == 3:
            if (history[0] == 'raise') and (history[1] == 'raise') and (history[2] == 'raise'):
                history_bucket = "vs_4bet"
            else:
                history_bucket = "vs_3bet"
        else:
            history_bucket = "vs_4bet"

        # ---------- betting size bucket ----------
        # Assumes there is some amount currently being faced.
        # Best case: state.to_call and blind size available.
        to_call = max(state.bets[0], state.bets[1]) - min(state.bets[0], state.bets[1])
        bb = max(state.blinds_or_straddles)

        # Size in big blinds
        size_bb = to_call / bb if bb > 0 else to_call

        if history_bucket == "limped":
            size_bucket = "limp"
        elif size_bb <= 2.0 and to_call > 0:
            size_bucket = "min_open"
        elif 2.0 < size_bb <= 2.75:
            size_bucket = "2.5x_open"
        elif 2.75 < size_bb < 6.0:
            size_bucket = "large_open"
        elif 6.0 <= size_bb < 10.0:
            size_bucket = "small_3bet"
        elif 10.0 <= size_bb < 25.0:
            size_bucket = "large_3bet"
        elif size_bb >= 25.0:
            size_bucket = "jam"
        else:
            size_bucket = "limp"

        return (
            hand_bucket,
            # position_bucket,
            stack_bucket,
            history_bucket,
            size_bucket,
        )
        


"""
---------------- Preflop
hand bucket
    premium pairs (A-J)
    medium pairs (10-8)
    small pairs (7-2)
    premium suited broadways (KQs - J10s)
    premium offsuit broadways (KQo)
    premium suited aces (AKs-A10s)
    weak suited aces (A9s-A2s)
    premium offsuit aces (AKo-AQo)
    suited connectors 
    suited gappers 
    weak offsuit broadways (KJo-J10o)
    trash offsuit hands (everything else)

position bucket
    SB
    BB
    BTN and other pos (for multiway, probably not using for prototype)

effective stack size (constrain to 100bb+ scenarios)
    0-20bb (short)
    20-50bb (medium)
    50-100bb (deep)
    100bb+ (very deep)

betting history (action / facing 3 bet / facing open)
    root
    open
    vs_open
    vs_3bet
    vs_4bet
    limped
    limp_raise

betting size
    limp
    min open
    2.5x open
    large open
    small 3-bet
    large 3-bet
    jam

----------------- Flop

HS
EHS
EHS²
maybe one or two draw flags

Turn

Same as flop, but maybe lighter-weight if speed is an issue

River

Usually HS alone is much more informative, since there are no future streets left
and EHS basically collapses toward current hand strength.
"""