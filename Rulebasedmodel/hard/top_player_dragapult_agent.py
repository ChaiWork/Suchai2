"""
Top Player Dragapult ex + Drakloak Engine Rule-Based Agent (Pro Version).
Ground Truth: Matches 92292731, 92300062, 92313203, 92314212 (Sixth Sense, LiamK, LumenLiquidity).

Real Decision Fingerprint (extracted from match JSONs):
  Top cards:  Dreepy(119), Dragapult ex(121), Drakloak(120), Munkidori(112),
              Budew(235), Crushing Hammer(1120), Poffin(1086), Lillie(1227),
              Ultra Ball(1121), Boss's Orders(1182), Crispin(1198),
              Night Stretcher(1097), Fezandipiti ex(140)
  Top attacks: Dragon Breath/153 (393x), Phantom Dive/154 (278x),
               Reconnaissance/323 (188x), Adrenaline Brains/141 (42x)
Strategy: Build Drakloak draw engine → Crispin 1-turn energy → Phantom Dive 200+60 bench snipe
"""

# ── Card IDs (verified from match JSON frequency counts) ──────────────────────
DREEPY_ID          = 119   # Dreepy
DRAKLOAK_ID        = 120   # Drakloak  [Reconnaissance: draw 2 pick 1]
DRAGAPULT_EX_ID    = 121   # Dragapult ex [Phantom Dive: 200+60 bench]
MUNKIDORI_ID       = 112   # Munkidori [Adrenaline Brains: damage transfer]
FEZANDIPITI_EX_ID  = 140   # Fezandipiti ex [Flip the Script: draw recovery]
BUDEW_ID           = 235   # Budew [pivot / fodder]

UNFAIR_STAMP_ID    = 1080  # Unfair Stamp ACE SPEC [hand reset to 2]
POFFIN_ID          = 1086  # Buddy-Buddy Poffin [bench 2x basics]
NIGHT_STRETCHER_ID = 1097  # Night Stretcher [retrieve from discard]
CRUSHING_HAMMER_ID = 1120  # Crushing Hammer [energy removal]
ULTRA_BALL_ID      = 1121  # Ultra Ball [search any Pokemon]
POKE_PAD_ID        = 1152  # Poké Pad [search item]
BOSS_ORDERS_ID     = 1182  # Boss's Orders [gust benched target]
CRISPIN_ID         = 1198  # Crispin [tutor Fire+Psychic directly to Pokemon]
LILLIES_DETERM_ID  = 1227  # Lillie's Determination [hand refresh]

# ── Attack IDs (verified from match JSON) ─────────────────────────────────────
DRAGON_BREATH_ID   = 153   # Dragon Breath (100 damage) - early game
PHANTOM_DIVE_ID    = 154   # Phantom Dive (200 active + 60 bench snipe) - win condition
RECONNAISSANCE_ID  = 323   # Reconnaissance (draw 2 pick 1) - Drakloak ability
ADRENALINE_ID      = 141   # Adrenaline Brains - Munkidori damage transfer

DRAGAPULT_LINE     = frozenset({DREEPY_ID, DRAKLOAK_ID, DRAGAPULT_EX_ID})
LOW_HP_TARGETS     = frozenset({119, 120, 235, 112, 140})  # easy prize targets on bench


def _get(opt, key, default=None):
    if isinstance(opt, dict):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _ctx(obs):
    """Extract game state context for smart decision making."""
    ctx = {
        "turn": 1, "my_prizes": 6, "opp_prizes": 6,
        "my_active_id": -1, "my_active_energy": 0, "my_active_hp": 320,
        "opp_active_id": -1, "opp_active_energy": 0, "opp_active_hp": 200,
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
            ctx["my_active_hp"]     = _get(ac, "hp", 320)
            energies = _get(ac, "energyCards", [])
            ctx["my_active_energy"] = len(energies) if isinstance(energies, list) else 0

    opp_active_list = _get(opp, "active", [])
    if opp_active_list:
        ac = opp_active_list[0]
        if ac:
            ctx["opp_active_id"]     = _get(ac, "cardId", -1)
            ctx["opp_active_hp"]     = _get(ac, "hp", 200)
            energies = _get(ac, "energyCards", [])
            ctx["opp_active_energy"] = len(energies) if isinstance(energies, list) else 0

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
    Pro Dragapult ex Rule-Based Agent.
    Priority: Phantom Dive KO > Evolve to Dragapult > Crispin acceleration >
              Poffin/Dreepy bench setup > Unfair Stamp disruption > Boss gust >
              Drakloak evolution > Night Stretcher recovery > Crushing Hammer
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
    turn        = ctx["turn"]
    my_prizes   = ctx["my_prizes"]
    opp_prizes  = ctx["opp_prizes"]
    active_id   = ctx["my_active_id"]
    bench_ids   = ctx["my_bench_ids"]
    opp_bench   = ctx["opp_bench_ids"]
    hand_ids    = ctx["my_hand_ids"]
    active_en   = ctx["my_active_energy"]
    active_hp   = ctx["my_active_hp"]

    early_game  = turn <= 3 or my_prizes >= 5
    close_game  = my_prizes <= 2 or opp_prizes <= 2
    # We have Dragapult ex on bench or as active
    has_dragapult = DRAGAPULT_EX_ID in bench_ids or active_id == DRAGAPULT_EX_ID
    has_drakloak  = DRAKLOAK_ID in bench_ids or active_id == DRAKLOAK_ID
    # How many Dreepy/Drakloak do we have set up (draw engine depth)
    engine_depth  = sum(1 for b in bench_ids if b in DRAGAPULT_LINE)
    opp_has_low_hp_bench = any(b in LOW_HP_TARGETS for b in opp_bench)

    best_idx   = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score    = 0.0
        opt_type = _get(opt, "type", 0)
        card_id  = _get(opt, "cardId", -1)
        attack_id = _get(opt, "attackId", -1)

        # ─── PRIORITY 1: ATTACK ────────────────────────────────────────────────
        if opt_type in (13, "Attack"):
            if attack_id == PHANTOM_DIVE_ID:
                # Phantom Dive = win condition. Always prefer.
                score = 10000.0
                if opp_prizes <= 2:
                    score += 5000.0  # Closing KO bonus
            elif attack_id == DRAGON_BREATH_ID:
                # Dragon Breath early game = deal 100, still good
                score = 5000.0 if active_en >= 1 else 2000.0
            elif attack_id == RECONNAISSANCE_ID:
                # Drakloak Reconnaissance: draw 2 pick 1 — treat like an attack
                score = 4000.0
            elif attack_id == ADRENALINE_ID:
                # Munkidori ability: redirect damage
                score = 3500.0
            else:
                score = 1000.0

        # ─── PRIORITY 2: EVOLVE ────────────────────────────────────────────────
        elif opt_type in (9, 12, "Evolve"):
            if card_id == DRAGAPULT_EX_ID:
                # Evolve Drakloak → Dragapult ex: unlock Phantom Dive
                score = 9000.0
            elif card_id == DRAKLOAK_ID:
                # Evolve Dreepy → Drakloak: unlock Reconnaissance draw engine
                score = 8000.0 if engine_depth < 2 else 7000.0

        # ─── PRIORITY 3: TRAINERS / SUPPORTERS ────────────────────────────────
        elif opt_type in (7, "Play"):
            if card_id == CRISPIN_ID:
                # Crispin: Tutor Fire + Psychic energy directly onto Dragapult ex (1-turn setup)
                if has_dragapult:
                    score = 8500.0  # Critical: enables Phantom Dive next turn
                else:
                    score = 5000.0

            elif card_id == UNFAIR_STAMP_ID:
                # Unfair Stamp: collapse opponent's hand to 2 after a KO
                if opp_prizes < 6:  # Opponent has taken prizes = KO happened
                    score = 8000.0  # Maximum disruption timing
                else:
                    score = 4000.0

            elif card_id == POFFIN_ID:
                # Buddy-Buddy Poffin: put 2 basics from deck to bench
                if len(bench_ids) < 3:
                    score = 7500.0  # Build engine early
                elif len(bench_ids) < 5:
                    score = 5000.0
                else:
                    score = 1500.0  # Bench is full, less valuable

            elif card_id == BOSS_ORDERS_ID:
                # Boss's Orders: gust low-HP bench target for free KO
                if opp_has_low_hp_bench:
                    score = 7000.0  # Gust Dreepy/Munkidori for easy prize
                elif close_game:
                    score = 6500.0
                else:
                    score = 4000.0

            elif card_id == DREEPY_ID:
                # Place Dreepy on bench = raw Dragapult line setup
                if len(bench_ids) < 3:
                    score = 7000.0
                elif engine_depth < 3:
                    score = 5500.0
                else:
                    score = 2000.0

            elif card_id == ULTRA_BALL_ID:
                # Search for Dragapult line or Munkidori
                no_drakloak_in_hand = DRAKLOAK_ID not in hand_ids and DRAGAPULT_EX_ID not in hand_ids
                if no_drakloak_in_hand and early_game:
                    score = 6500.0
                else:
                    score = 4000.0

            elif card_id == MUNKIDORI_ID:
                # Place Munkidori to bench (supports damage transfer)
                if MUNKIDORI_ID not in bench_ids:
                    score = 6000.0
                else:
                    score = 2000.0

            elif card_id == NIGHT_STRETCHER_ID:
                # Retrieve Pokemon/Energy from discard
                score = 5500.0 if my_prizes <= 4 else 3500.0

            elif card_id == FEZANDIPITI_EX_ID:
                # Fezandipiti ex: Flip the Script draw recovery after KO
                score = 5000.0 if len(bench_ids) < 5 else 2500.0

            elif card_id == LILLIES_DETERM_ID:
                # Lillie's Determination: draw up to 8 cards
                score = 5000.0 if ctx["my_hand_len"] < 4 else 3000.0

            elif card_id == POKE_PAD_ID:
                # Poké Pad: search item card
                score = 4500.0

            elif card_id == CRUSHING_HAMMER_ID:
                # Remove opponent energy — useful in mid/close game
                score = 4000.0 if opp_bench else 2000.0

        # ─── PRIORITY 4: ENERGY ATTACHMENT ─────────────────────────────────────
        elif opt_type in (8, "Attach"):
            if active_id == DRAGAPULT_EX_ID:
                score = 6000.0  # Powering up Dragapult ex
            elif DRAGAPULT_EX_ID in bench_ids:
                score = 5000.0  # Attach to bench Dragapult ex
            else:
                score = 3000.0

        # ─── PRIORITY 5: RETREAT ────────────────────────────────────────────────
        elif opt_type in (5, "Retreat"):
            if active_id == DRAGAPULT_EX_ID and active_en >= 2 and active_hp > 50:
                # Do NOT retreat a powered Dragapult ex — attack instead
                score = -9000.0
            elif active_id in (BUDEW_ID,) and active_hp < 50:
                # Pivot out damaged basics
                score = 3000.0
            else:
                score = 500.0

        # ─── PRIORITY 6: ABILITY ────────────────────────────────────────────────
        elif opt_type in (17, "Ability"):
            score = 7000.0  # Reconnaissance draw or Munkidori damage transfer

        if score > best_score:
            best_score = score
            best_idx   = idx

    return [best_idx]
