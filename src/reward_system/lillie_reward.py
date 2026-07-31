"""
Lillie's Clefairy ex + Togekiss Specific Strategic Reward Engine.
Implements opening rewards, evolution rewards, energy acceleration rewards, supporter & item rewards,
stadium timing, recovery rewards, tactical KO rewards, and matchup-specific reward signals for:
- Dragapult ex
- Crustle
- Lucario
- Grimmsnarl ex
"""

# Card ID Constants
CLEFAIRY_EX_ID       = 272   # Lillie's Clefairy ex
TOGEKISS_ID          = 214   # Togekiss
TOGEPI_ID            = 959   # Togepi
TOGETIC_ID           = 960   # Togetic
SMOOCHUM_ID          = 183   # Smoochum
LATIAS_EX_ID         = 184   # Latias ex

TELEPATHIC_ENERGY_ID = 19    # Telepathic Psychic Energy
BASIC_PSYCHIC_ID     = 5     # Basic Psychic Energy

MYSTERY_GARDEN_ID    = 1263  # Mystery Garden
WONDROUS_PATCH_ID    = 1146  # Wondrous Patch
LILLIES_PEARL_ID     = 1172  # Lillie's Pearl
RARE_CANDY_ID        = 1079  # Rare Candy
ULTRA_BALL_ID        = 1121  # Ultra Ball
POKE_PAD_ID          = 1152  # Poké Pad
NIGHT_STRETCHER_ID   = 1097  # Night Stretcher
AIR_BALLOON_ID       = 1174  # Air Balloon
SWITCH_ID            = 1123  # Switch
UNFAIR_STAMP_ID      = 1080  # Unfair Stamp
ACCOMPANYING_FLUTE_ID= 1091  # Accompanying Flute
POKEMON_CATCHER_ID   = 1124  # Pokémon Catcher

COLRESS_TENACITY_ID  = 1194  # Colress's Tenacity
HILDA_ID             = 1225  # Hilda
LILLIES_DETERM_ID    = 1227  # Lillie's Determination
MORTYS_CONVICTION_ID = 1187  # Morty's Conviction
BOSS_ORDERS_ID       = 1182  # Boss's Orders

# Matchup Evolving Pre-Evolution Card ID Sets
DRAGAPULT_LINE_IDS   = {336, 337, 338}   # Dreepy, Drakloak, Dragapult ex
CRUSTLE_LINE_IDS     = {344, 345}        # Dwebble, Crustle
LUCARIO_LINE_IDS     = {447, 448}        # Riolu, Lucario
GRIMMSNARL_LINE_IDS  = {646, 647, 648}   # Impidimp, Morgrem, Grimmsnarl ex


def calculate_lillie_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
    """
    Computes bounded strategic reward modifier r_strategic in range [-0.25, +0.25] for Lillie's Clefairy ex + Togekiss deck.
    """
    r_strategic = 0.0
    opp_lower = (opponent_name or "").lower()

    # Matchup Flags
    is_dragapult = "dragapult" in opp_lower
    is_crustle   = "crustle" in opp_lower
    is_lucario   = "lucario" in opp_lower
    is_grimmsnarl= "grimmsnarl" in opp_lower
    is_tank      = is_crustle or "abomasnow" in opp_lower or "kangaskhan" in opp_lower

    # -------------------------------------------------------------------------
    # 1. PLAY ACTIONS (action_type == 7)
    # -------------------------------------------------------------------------
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)

        # --- A. Opening Setup & Basic Benching ---
        if played_id in (CLEFAIRY_EX_ID, TOGEPI_ID, SMOOCHUM_ID, LATIAS_EX_ID):
            # Penalty for over-benching against Dragapult spread threat
            if is_dragapult and post.get("bench_size", 0) > 3:
                r_strategic -= 0.10
            else:
                r_strategic += 0.15

        # --- B. Supporter Play Rewards ---
        elif played_id == COLRESS_TENACITY_ID:
            r_strategic += 0.20  # Ideal Stadium + Energy tutor

        elif played_id == HILDA_ID:
            r_strategic += 0.20  # Evolution + Energy tutor

        elif played_id in (LILLIES_DETERM_ID, MORTYS_CONVICTION_ID):
            r_strategic += 0.12  # Hand refresh

        elif played_id == BOSS_ORDERS_ID:
            r_strategic += 0.18  # Gust disruption

        # --- C. Item Card Play Rewards & Penalties ---
        elif played_id == RARE_CANDY_ID:
            r_strategic += 0.25  # Fast Stage 2 Togekiss shortcut

        elif played_id == ULTRA_BALL_ID:
            r_strategic += 0.15  # Unrestricted Pokémon search

        elif played_id == WONDROUS_PATCH_ID:
            r_strategic += 0.25  # Discard energy acceleration to bench/active

        elif played_id == MYSTERY_GARDEN_ID:
            r_strategic += 0.18  # Stadium play & turn draw engine

        elif played_id == ACCOMPANYING_FLUTE_ID:
            # Accompanying Flute fills opponent's bench, boosting Clefairy ex Full Moon Rondo damage (+20/benched)
            r_strategic += 0.25 if is_tank else 0.20

        elif played_id == LILLIES_PEARL_ID:
            r_strategic += 0.25 if is_tank else 0.18  # Clefairy ex Tool equip (+30 damage)

        elif played_id == NIGHT_STRETCHER_ID:
            r_strategic += 0.12  # Recovery from discard

        elif played_id == POKE_PAD_ID:
            r_strategic += 0.15  # Supporter recycle to deck for draw engine continuity

        elif played_id == UNFAIR_STAMP_ID:
            r_strategic += 0.15  # Mid/Late hand disruption

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT ACTIONS (action_type == 8)
    # -------------------------------------------------------------------------
    elif action_type == 8:
        attached_id = pre.get("attached_card_id", -1)
        if attached_id == TELEPATHIC_ENERGY_ID:
            r_strategic += 0.22  # Attachment + 2 Basic {P} benching
        else:
            r_strategic += 0.10

    # -------------------------------------------------------------------------
    # 3. EVOLUTION ACTIONS (action_type == 9)
    # -------------------------------------------------------------------------
    elif action_type == 9:
        evolved_id = post.get("evolved_card_id", -1)
        if evolved_id == TOGEKISS_ID:
            r_strategic += 0.25
        elif evolved_id == TOGETIC_ID:
            r_strategic += 0.12

    # -------------------------------------------------------------------------
    # 4. MATCHUP-SPECIFIC TARGET KO REWARDS
    # -------------------------------------------------------------------------
    opp_pk_lost = pre.get("opp_pokemon", 0) - post.get("opp_pokemon", 0)
    if opp_pk_lost > 0:
        pre_opp_b = pre.get("opp_bench_ids", [])
        post_opp_b = post.get("opp_bench_ids", [])
        for cid in pre_opp_b:
            if cid not in post_opp_b:
                if is_dragapult and cid in (336, 337):
                    r_strategic += 0.35  # Knocked out Dreepy/Drakloak before Dragapult evolution
                elif is_lucario and cid == 447:
                    r_strategic += 0.35  # Knocked out Riolu before Lucario evolution
                elif is_grimmsnarl and cid in (646, 647, 112):
                    r_strategic += 0.35  # Knocked out Impidimp/Morgrem/Munkidori support
                elif is_crustle and cid in (344, 345):
                    r_strategic += 0.35

    # -------------------------------------------------------------------------
    # 5. BENCH DENSITY & PRE-ATTACK CARD UTILIZATION REWARDS
    # -------------------------------------------------------------------------
    my_bench_ids = post.get("my_bench_ids", [])
    clefairy_count = sum(1 for cid in my_bench_ids if cid == CLEFAIRY_EX_ID)
    if clefairy_count >= 2:
        r_strategic += 0.10

    # Reward playing hand cards (Supporters, Items, Energy, Evolutions) to improve setup prior to attacking
    if action_type in (7, 8, 9):
        r_strategic += 0.05

    # Bounded in range [-0.25, +0.25]
    r_strategic = max(-0.25, min(0.25, r_strategic))
    return r_strategic
