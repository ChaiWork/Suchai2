"""
Top Player Mega Lucario ex Fighting Burst Rule-Based Agent (Pro Version).
Ground Truth: Matches 92307565, 92310379 (Majkel1337, Luca).

Real Decision Fingerprint (extracted from match JSONs):
  Top cards:  Fighting Energy(6), Drakloak(120), Psychic Energy(5),
              Dragapult ex(121), Budew(235), Dreepy(119), Boss's Orders(1182),
              Lillie(1227), Crispin(1198), Ultra Ball(1121), Poké Pad(1152),
              Mega Lucario ex(676), Lucario ex(675), Lucario(674), Riolu(673)
  Top attacks: Mega Lucario ex atk(982=176x), Reconnaissance/323(108x),
               Mega Lucario ex atk-2(983=96x), Phantom Dive/154(79x),
               Dragon Breath/153(57x), Adrenaline Brains/141(51x)

Strategy: Fill bench with Riolu/Lucario → evolve Mega Lucario ex →
          use Fighting energy + Crispin for burst Fighting attacks (200-260 dmg).
          Includes Dragapult ex as secondary attacker / bench sniper.
"""

# ── Card IDs (verified from match JSON frequency counts) ──────────────────────
RIOLU_ID           = 673   # Riolu (Lucario base)
LUCARIO_ID         = 674   # Lucario
LUCARIO_EX_ID      = 675   # Lucario ex
MEGA_LUCARIO_EX_ID = 676   # Mega Lucario ex [flagship attacker]
DREEPY_ID          = 119   # Dreepy (Dragapult secondary line)
DRAKLOAK_ID        = 120   # Drakloak
DRAGAPULT_EX_ID    = 121   # Dragapult ex [Phantom Dive secondary]
MUNKIDORI_ID       = 112   # Munkidori [Adrenaline Brains]
BUDEW_ID           = 235   # Budew [pivot]
FEZANDIPITI_EX_ID  = 140   # Fezandipiti ex [draw]

ULTRA_BALL_ID      = 1121  # Ultra Ball
POKE_PAD_ID        = 1152  # Poké Pad [search item]
BOSS_ORDERS_ID     = 1182  # Boss's Orders [gust]
CRISPIN_ID         = 1198  # Crispin [energy acceleration]
LILLIES_DETERM_ID  = 1227  # Lillie's Determination
CRUSHING_HAMMER_ID = 1120  # Crushing Hammer

# Unknown from match data — likely supporter (92310379 shows cardId=1213)
UNKNOWN_SUPP_1213  = 1213  # Unknown supporter — treat as generic draw

# ── Attack IDs (verified from match JSON) ─────────────────────────────────────
MEGA_LUCARIO_ATK1_ID  = 982   # Mega Lucario ex attack 1 (176x in matches) — main attack
MEGA_LUCARIO_ATK2_ID  = 983   # Mega Lucario ex attack 2 (96x) — stronger burst
LUCARIO_EX_ATK_ID     = 980   # Lucario ex attack (31x)
RECON_ID              = 323   # Reconnaissance (Drakloak draw 2 pick 1)
PHANTOM_DIVE_ID       = 154   # Phantom Dive (200+60 bench snipe)
DRAGON_BREATH_ID      = 153   # Dragon Breath (100)
ADRENALINE_ID         = 141   # Adrenaline Brains (Munkidori damage transfer)

FIGHTING_ENERGY_ID = 6   # Fighting Energy
LUCARIO_LINE       = frozenset({RIOLU_ID, LUCARIO_ID, LUCARIO_EX_ID, MEGA_LUCARIO_EX_ID})
DRAGAPULT_LINE     = frozenset({DREEPY_ID, DRAKLOAK_ID, DRAGAPULT_EX_ID})


def _get(opt, key, default=None):
    if isinstance(opt, dict):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _ctx(obs):
    ctx = {
        "turn": 1, "my_prizes": 6, "opp_prizes": 6,
        "my_active_id": -1, "my_active_energy": 0, "my_active_hp": 300,
        "opp_active_id": -1, "opp_active_hp": 200,
        "my_bench_ids": [], "opp_bench_ids": [],
        "my_hand_ids": [], "my_hand_len": 0,
    }
    current = obs.get("current") if isinstance(obs, dict) else None
    if current is None:
        return ctx
    ctx["turn"] = _get(current, "turn", 1)
    yi = _get(current, "yourIndex", 0)
    players = _get(current, "players", [])
    if not players or len(players) < 2:
        return ctx

    me = players[yi]
    opp = players[1 - yi]

    prizes_my  = _get(me,  "prize", [])
    prizes_opp = _get(opp, "prize", [])
    ctx["my_prizes"]  = len(prizes_my)  if isinstance(prizes_my,  list) else 6
    ctx["opp_prizes"] = len(prizes_opp) if isinstance(prizes_opp, list) else 6

    my_active_list = _get(me, "active", [])
    if my_active_list:
        ac = my_active_list[0]
        if ac:
            ctx["my_active_id"]     = _get(ac, "cardId", -1)
            ctx["my_active_hp"]     = _get(ac, "hp", 300)
            energies = _get(ac, "energyCards", [])
            ctx["my_active_energy"] = len(energies) if isinstance(energies, list) else 0

    opp_active_list = _get(opp, "active", [])
    if opp_active_list:
        ac = opp_active_list[0]
        if ac:
            ctx["opp_active_id"] = _get(ac, "cardId", -1)
            ctx["opp_active_hp"] = _get(ac, "hp", 200)

    my_bench  = _get(me,  "bench", [])
    opp_bench = _get(opp, "bench", [])
    ctx["my_bench_ids"]  = [_get(b, "cardId", -1) for b in my_bench  if b] if isinstance(my_bench,  list) else []
    ctx["opp_bench_ids"] = [_get(b, "cardId", -1) for b in opp_bench if b] if isinstance(opp_bench, list) else []

    my_hand = _get(me, "hand", [])
    if isinstance(my_hand, list):
        ctx["my_hand_ids"] = [_get(h, "cardId", -1) for h in my_hand if h]
        ctx["my_hand_len"] = len(ctx["my_hand_ids"])

    return ctx


def agent(obs):
    """
    Pro Mega Lucario ex Rule-Based Agent.
    Priority: Mega Lucario ex burst attack > evolve chain > Crispin energy >
              Reconnaissance draw > bench setup > Boss's Orders gust.
    """
    select_info = obs.get("select", {}) if isinstance(obs, dict) else {}
    if not select_info or not isinstance(select_info, dict):
        return []
    options = select_info.get("option", [])
    if not options:
        return []
    if len(options) == 1:
        return [0]

    ctx = _ctx(obs)
    turn       = ctx["turn"]
    my_prizes  = ctx["my_prizes"]
    opp_prizes = ctx["opp_prizes"]
    active_id  = ctx["my_active_id"]
    bench_ids  = ctx["my_bench_ids"]
    opp_bench  = ctx["opp_bench_ids"]
    hand_ids   = ctx["my_hand_ids"]
    active_en  = ctx["my_active_energy"]
    active_hp  = ctx["my_active_hp"]

    early_game   = turn <= 3 or my_prizes >= 5
    close_game   = my_prizes <= 2 or opp_prizes <= 2
    has_mega_luc = MEGA_LUCARIO_EX_ID in bench_ids or active_id == MEGA_LUCARIO_EX_ID
    has_luc_ex   = LUCARIO_EX_ID in bench_ids or active_id == LUCARIO_EX_ID
    lucario_ct   = sum(1 for b in bench_ids if b in LUCARIO_LINE)
    bench_full   = len(bench_ids) >= 4

    opp_low_hp   = any(b in (119, 120, 673, 235, 112, 140) for b in opp_bench)

    best_idx   = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score     = 0.0
        opt_type  = _get(opt, "type", 0)
        card_id   = _get(opt, "cardId", -1)
        attack_id = _get(opt, "attackId", -1)

        # ─── PRIORITY 1: ATTACK ────────────────────────────────────────────────
        if opt_type in (13, "Attack"):
            if attack_id == MEGA_LUCARIO_ATK2_ID:
                # Strongest Mega Lucario ex attack — burst finisher
                score = 10000.0 if close_game else 9000.0
            elif attack_id == MEGA_LUCARIO_ATK1_ID:
                # Main Mega Lucario ex attack (most used: 176x)
                score = 9500.0
            elif attack_id == PHANTOM_DIVE_ID:
                # Dragapult ex Phantom Dive — secondary win condition
                score = 8500.0 if opp_low_hp else 7000.0
            elif attack_id == LUCARIO_EX_ATK_ID:
                score = 7000.0
            elif attack_id == RECON_ID:
                # Drakloak draw 2 pick 1
                score = 6000.0
            elif attack_id == DRAGON_BREATH_ID:
                score = 5000.0
            elif attack_id == ADRENALINE_ID:
                score = 4500.0
            else:
                score = 2000.0

        # ─── PRIORITY 2: ABILITY ────────────────────────────────────────────────
        elif opt_type in (17, "Ability"):
            score = 8500.0

        # ─── PRIORITY 3: EVOLVE ────────────────────────────────────────────────
        elif opt_type in (9, 12, "Evolve"):
            if card_id == MEGA_LUCARIO_EX_ID:
                score = 9000.0  # Evolve to Mega Lucario ex
            elif card_id == LUCARIO_EX_ID:
                score = 8000.0  # Evolve to Lucario ex
            elif card_id == LUCARIO_ID:
                score = 7000.0
            elif card_id == DRAGAPULT_EX_ID:
                score = 8500.0  # Secondary attacker: Phantom Dive
            elif card_id == DRAKLOAK_ID:
                score = 7500.0  # Draw engine

        # ─── PRIORITY 4: TRAINERS / SUPPORTERS ─────────────────────────────────
        elif opt_type in (7, "Play"):
            if card_id == CRISPIN_ID:
                # Energy acceleration onto Mega Lucario ex
                if has_mega_luc or has_luc_ex:
                    score = 9000.0
                else:
                    score = 5000.0

            elif card_id == BOSS_ORDERS_ID:
                # Gust low-HP bench target for prize
                if opp_low_hp:
                    score = 8000.0
                elif close_game:
                    score = 7000.0
                else:
                    score = 4000.0

            elif card_id == RIOLU_ID:
                # Place Riolu — start Lucario evolution line
                if lucario_ct < 2:
                    score = 7500.0
                elif not bench_full:
                    score = 5000.0
                else:
                    score = 1500.0

            elif card_id == ULTRA_BALL_ID:
                no_mega_hand = MEGA_LUCARIO_EX_ID not in hand_ids and LUCARIO_EX_ID not in hand_ids
                score = 7000.0 if (no_mega_hand and early_game) else 4000.0

            elif card_id == DREEPY_ID:
                # Place Dreepy for secondary Dragapult line
                if not bench_full and lucario_ct >= 1:
                    score = 6500.0
                else:
                    score = 2500.0

            elif card_id == MUNKIDORI_ID:
                if MUNKIDORI_ID not in bench_ids and not bench_full:
                    score = 6000.0
                else:
                    score = 1500.0

            elif card_id == LILLIES_DETERM_ID:
                score = 5500.0 if ctx["my_hand_len"] < 4 else 3000.0

            elif card_id == UNKNOWN_SUPP_1213:
                score = 5000.0  # Unknown supporter — treat as draw

            elif card_id == CRUSHING_HAMMER_ID:
                score = 4000.0 if opp_bench else 2000.0

            elif card_id == FEZANDIPITI_EX_ID:
                score = 4500.0 if not bench_full else 2000.0

            elif card_id == POKE_PAD_ID:
                score = 4000.0

        # ─── PRIORITY 5: ENERGY ATTACHMENT ─────────────────────────────────────
        elif opt_type in (8, "Attach"):
            if active_id == MEGA_LUCARIO_EX_ID:
                score = 7000.0
            elif MEGA_LUCARIO_EX_ID in bench_ids:
                score = 6500.0
            elif active_id == LUCARIO_EX_ID or LUCARIO_EX_ID in bench_ids:
                score = 5500.0
            else:
                score = 3000.0

        # ─── PRIORITY 6: RETREAT ─────────────────────────────────────────────
        elif opt_type in (5, "Retreat"):
            if active_id == MEGA_LUCARIO_EX_ID and active_en >= 2 and active_hp > 80:
                score = -9000.0  # Never retreat powered Mega Lucario ex
            elif active_id in (RIOLU_ID, BUDEW_ID) and active_hp < 40:
                score = 3000.0
            else:
                score = 500.0

        if score > best_score:
            best_score = score
            best_idx   = idx

    return [best_idx]
