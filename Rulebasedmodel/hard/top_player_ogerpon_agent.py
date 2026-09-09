"""
Top Player Teal Mask Ogerpon ex + Meganium / Forest of Vitality Rule-Based Agent (Pro Version).
Ground Truth: Matches 92304758, 92308710, 92315100 (Oshbocker, ANDPAD kaggler team, Dipam).

Real Decision Fingerprint (extracted from match JSONs):
  Top cards:  Grass Energy(1), Teal Mask Ogerpon ex(150), Chikorita(96),
              Budew(235), Ultra Ball(1121), Psychic Energy(5), Fire Energy(2),
              Dreepy(119), Boss's Orders(1182), Lillie(1227), Meowth ex(1071),
              Bug Catching Set(1094), Fezandipiti ex(140)
  Top attacks: Teal Dance/195 (377x), Reconnaissance/323 (270x),
               Dragon Breath/153 (114x), Phantom Dive/154 (105x)

Strategy: Teal Dance energy acceleration to flood Grass energies → sweep with
          Ogerpon's powered grass attacks. Meganium adds Grass energy from deck.
"""

# ── Card IDs (verified from match JSON frequency counts) ──────────────────────
CHIKORITA_ID       = 96    # Chikorita (Meganium line)
BAYLEEF_ID         = 97    # Bayleef
MEGANIUM_ID        = 154   # Meganium [Fragrant Herb: search Grass energy/Pokemon]
OGERPON_EX_ID      = 142   # Ogerpon ex (base)
TEAL_OGERPON_ID    = 150   # Teal Mask Ogerpon ex [Teal Dance: attach Grass from hand]
MEOWTH_EX_ID       = 1071  # Meowth ex [Jealous Heart: draw acceleration]
FEZANDIPITI_EX_ID  = 140   # Fezandipiti ex [Flip the Script]
BUDEW_ID           = 235   # Budew [pivot]

ULTRA_BALL_ID      = 1121  # Ultra Ball [search any Pokemon]
POKE_PAD_ID        = 1152  # Poké Pad
BUG_CATCHING_SET   = 1094  # Bug Catching Set [search Grass Pokemon or Energy]
BOSS_ORDERS_ID     = 1182  # Boss's Orders [gust bench target]
LILLIES_DETERM_ID  = 1227  # Lillie's Determination [draw]
FOREST_VITALITY_ID = 1258  # Forest of Vitality [heal 30 after attach]
CRUSHING_HAMMER_ID = 1120  # Crushing Hammer

# ── Attack IDs (verified from match JSON) ─────────────────────────────────────
TEAL_DANCE_ID      = 195   # Teal Dance (attach Grass energy from hand + draw) = 377x
RECON_ID           = 323   # Reconnaissance (Drakloak draw 2 pick 1) = 270x
DRAGON_BREATH_ID   = 153   # Dragon Breath (100) = 114x
PHANTOM_DIVE_ID    = 154   # Phantom Dive (200+60 bench) = 105x

GRASS_ENERGY_ID    = 1     # Grass Energy
OGERPON_LINE       = frozenset({96, 97, 142, 150})  # Ogerpon + Meganium line


def _get(opt, key, default=None):
    if isinstance(opt, dict):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _ctx(obs):
    ctx = {
        "turn": 1, "my_prizes": 6, "opp_prizes": 6,
        "my_active_id": -1, "my_active_energy": 0, "my_active_hp": 340,
        "opp_active_id": -1, "opp_active_hp": 200,
        "my_bench_ids": [], "opp_bench_ids": [],
        "my_hand_ids": [], "my_hand_len": 0,
        "my_hand_grass": 0,
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
            ctx["my_active_hp"]     = _get(ac, "hp", 340)
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
        ctx["my_hand_ids"]   = [_get(h, "cardId", -1) for h in my_hand if h]
        ctx["my_hand_len"]   = len(ctx["my_hand_ids"])
        ctx["my_hand_grass"] = ctx["my_hand_ids"].count(GRASS_ENERGY_ID)

    return ctx


def agent(obs):
    """
    Pro Ogerpon ex Rule-Based Agent.
    Priority: Teal Dance energy flood → attack with full-power Ogerpon ex.
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
    turn         = ctx["turn"]
    my_prizes    = ctx["my_prizes"]
    opp_prizes   = ctx["opp_prizes"]
    active_id    = ctx["my_active_id"]
    bench_ids    = ctx["my_bench_ids"]
    opp_bench    = ctx["opp_bench_ids"]
    hand_ids     = ctx["my_hand_ids"]
    hand_grass   = ctx["my_hand_grass"]
    active_en    = ctx["my_active_energy"]
    active_hp    = ctx["my_active_hp"]

    early_game   = turn <= 3 or my_prizes >= 5
    close_game   = my_prizes <= 2 or opp_prizes <= 2
    has_ogerpon  = TEAL_OGERPON_ID in bench_ids or active_id == TEAL_OGERPON_ID
    bench_full   = len(bench_ids) >= 4
    # Ogerpon damage = 70 + 30 per Grass energy attached (so 3 energy = 160 damage)
    ogerpon_powered = active_en >= 3

    best_idx   = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score     = 0.0
        opt_type  = _get(opt, "type", 0)
        card_id   = _get(opt, "cardId", -1)
        attack_id = _get(opt, "attackId", -1)

        # ─── PRIORITY 1: ATTACK ────────────────────────────────────────────────
        if opt_type in (13, "Attack"):
            if attack_id == TEAL_DANCE_ID:
                # Teal Dance: attach Grass energy from hand + draw — core accelerator
                if hand_grass > 0:
                    score = 10000.0 + hand_grass * 1000.0  # More grass = more value
                else:
                    score = 5000.0  # Still use to draw
            elif attack_id == PHANTOM_DIVE_ID:
                score = 9000.0 if close_game else 7000.0
            elif attack_id == DRAGON_BREATH_ID:
                score = 6000.0
            elif attack_id == RECON_ID:
                score = 5500.0  # Drakloak draw
            else:
                score = 2000.0

        # ─── PRIORITY 2: ABILITY ────────────────────────────────────────────────
        elif opt_type in (17, "Ability"):
            # Teal Dance triggered via ability or Meganium's herb effect
            score = 9500.0

        # ─── PRIORITY 3: EVOLVE ────────────────────────────────────────────────
        elif opt_type in (9, 12, "Evolve"):
            if card_id == TEAL_OGERPON_ID:
                score = 8500.0  # Evolve to Teal Mask Ogerpon ex
            elif card_id == MEGANIUM_ID:
                score = 7500.0  # Evolve Bayleef to Meganium (energy search)
            elif card_id == BAYLEEF_ID:
                score = 7000.0

        # ─── PRIORITY 4: TRAINERS / SUPPORTERS ─────────────────────────────────
        elif opt_type in (7, "Play"):
            if card_id == FOREST_VITALITY_ID:
                # Heal 30 HP after energy attachment each turn
                score = 8000.0  # Always want this stadium up

            elif card_id == BUG_CATCHING_SET:
                # Search for Grass Pokemon or Grass Energy directly
                score = 7500.0 if not has_ogerpon else 5000.0

            elif card_id == ULTRA_BALL_ID:
                # Search Ogerpon ex or Meganium line
                no_ogerpon_hand = TEAL_OGERPON_ID not in hand_ids and OGERPON_EX_ID not in hand_ids
                score = 7000.0 if (no_ogerpon_hand and early_game) else 4000.0

            elif card_id == CHIKORITA_ID:
                # Bench Chikorita for Meganium evolution (draw engine)
                score = 7000.0 if not bench_full else 2000.0

            elif card_id == TEAL_OGERPON_ID:
                # Place Ogerpon ex on bench
                score = 7000.0 if TEAL_OGERPON_ID not in bench_ids else 2000.0

            elif card_id == BOSS_ORDERS_ID:
                # Gust low-HP bench Pokemon
                opp_vuln = any(b in (96, 235, 119, 120) for b in opp_bench)
                score = 7000.0 if opp_vuln else (5500.0 if close_game else 3500.0)

            elif card_id == MEOWTH_EX_ID:
                # Meowth ex: Jealous Heart = draw when opponent uses supporter
                score = 6000.0 if not bench_full else 2000.0

            elif card_id == LILLIES_DETERM_ID:
                score = 5500.0 if ctx["my_hand_len"] < 4 else 3000.0

            elif card_id == FEZANDIPITI_EX_ID:
                score = 5000.0 if not bench_full else 2000.0

            elif card_id == CRUSHING_HAMMER_ID:
                score = 4000.0 if opp_bench else 2000.0

            elif card_id == POKE_PAD_ID:
                score = 4000.0

        # ─── PRIORITY 5: ENERGY ATTACHMENT ─────────────────────────────────────
        elif opt_type in (8, "Attach"):
            if active_id == TEAL_OGERPON_ID:
                score = 7000.0  # Always attach Grass to active Ogerpon
            elif TEAL_OGERPON_ID in bench_ids:
                score = 6000.0
            else:
                score = 3000.0

        # ─── PRIORITY 6: RETREAT ─────────────────────────────────────────────
        elif opt_type in (5, "Retreat"):
            if active_id == TEAL_OGERPON_ID and ogerpon_powered and active_hp > 80:
                score = -9000.0  # Never retreat a powered Ogerpon ex
            elif active_id == BUDEW_ID and active_hp < 40:
                score = 3000.0
            else:
                score = 500.0

        if score > best_score:
            best_score = score
            best_idx   = idx

    return [best_idx]
