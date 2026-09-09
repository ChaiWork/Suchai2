"""
Top Player Marnie's Grimmsnarl ex + Munkidori Rule-Based Agent (Pro Version).
Ground Truth: Match 92309539 (Phil_Hellmuth / The Malaysia Peas).

Real Decision Fingerprint (extracted from match JSON 92309539):
  Top cards:  Darkness Energy(7), Marnie's Grimmsnarl ex(743), Munkidori(112),
              Grimmsnarl ex(648), Marnie's Morpeko(741), Impidimp(646),
              742(?), Psychic-type(66), Poké Pad(1152), misc(305),
              Poffin(1086), Unfair Stamp(1079/1080), Spikemuth Gym(1259),
              Morgrem(647), ?-supporter(1231), Petrel(1219), ?-supporter(1225),
              Fezandipiti ex(140)
  Top attacks: Marnie's Grimmsnarl ex attack(1072=136x), 937(37x), 936(22x)

Strategy: Evolve Impidimp → Morgrem → Grimmsnarl ex / Marnie's Grimmsnarl ex
          Use Munkidori (Adrenaline Brains) to route damage counters strategically.
          Spikemuth Gym boosts Darkness attacks.
          Main attack = 1072 (Marnie's Grimmsnarl ex: likely 180+ damage).
"""

# ── Card IDs (verified from match JSON 92309539) ─────────────────────────────
IMPIDIMP_ID             = 646   # Impidimp (Grimmsnarl base)
MORGREM_ID              = 647   # Morgrem (Stage 1)
GRIMMSNARL_EX_ID        = 648   # Grimmsnarl ex [Darkness burst]
MARNIE_GRIMMSNARL_EX_ID = 743   # Marnie's Grimmsnarl ex [flagship: 1072]
MARNIE_MORPEKO_ID       = 741   # Marnie's Morpeko [speed electric support]
MORPEKO_EVOLVE_ID       = 742   # Marnie's Morpeko evolution (?)
MUNKIDORI_ID            = 112   # Munkidori [Adrenaline Brains: damage transfer]
FEZANDIPITI_EX_ID       = 140   # Fezandipiti ex [Flip the Script draw]

POFFIN_ID          = 1086  # Buddy-Buddy Poffin [bench 2x basics]
UNFAIR_STAMP_ID    = 1079  # Unfair Stamp (seen as 1079 in this deck — Marnie version?)
UNFAIR_STAMP_ALT   = 1080  # Standard Unfair Stamp
POKE_PAD_ID        = 1152  # Poké Pad [search item]
SPIKEMUTH_GYM_ID   = 1259  # Spikemuth Gym [boost Darkness damage]
PETREL_ID          = 1219  # Team Rocket's Petrel [recycle cards from discard]
UNKNOWN_SUPP_1231  = 1231  # Unidentified supporter (treat as draw)
UNKNOWN_SUPP_1225  = 1225  # Unidentified supporter (treat as draw)
MISC_305           = 305   # Unknown item

# ── Attack IDs (verified from match JSON 92309539) ────────────────────────────
MARNIE_GRIMMSNARL_ATK1  = 1072  # Marnie's Grimmsnarl ex main attack (136x) — win condition
GRIMMSNARL_ATK2         = 937   # Secondary attack (37x)
GRIMMSNARL_ATK1         = 936   # Attack 1 (22x)
GRIMMSNARL_ATK3         = 934   # Rare attack
GRIMMSNARL_ATK4         = 935   # Rare attack

DARKNESS_ENERGY_ID = 7   # Darkness Energy
PSYCHIC_ENERGY_ID  = 5   # Psychic Energy (for Munkidori)
GRIMMSNARL_LINE    = frozenset({IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID, MARNIE_GRIMMSNARL_EX_ID})


def _get(opt, key, default=None):
    if isinstance(opt, dict):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _ctx(obs):
    ctx = {
        "turn": 1, "my_prizes": 6, "opp_prizes": 6,
        "my_active_id": -1, "my_active_energy": 0, "my_active_hp": 320,
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
            ctx["my_active_hp"]     = _get(ac, "hp", 320)
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
    Pro Marnie's Grimmsnarl ex Rule-Based Agent.
    Priority: Marnie's Grimmsnarl ex attack > evolve chain > Munkidori damage transfer >
              Spikemuth Gym > Poffin bench setup > Petrel recycle > Unfair Stamp
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
    grimsnarl_ct = sum(1 for b in bench_ids if b in GRIMMSNARL_LINE)
    has_marnie_grim = MARNIE_GRIMMSNARL_EX_ID in bench_ids or active_id == MARNIE_GRIMMSNARL_EX_ID
    has_munkidori   = MUNKIDORI_ID in bench_ids
    bench_full      = len(bench_ids) >= 4
    opp_low_hp      = any(b in (119, 120, 646, 673, 112, 235, 140) for b in opp_bench)

    best_idx   = 0
    best_score = -9999.0

    for idx, opt in enumerate(options):
        score     = 0.0
        opt_type  = _get(opt, "type", 0)
        card_id   = _get(opt, "cardId", -1)
        attack_id = _get(opt, "attackId", -1)

        # ─── PRIORITY 1: ATTACK ────────────────────────────────────────────────
        if opt_type in (13, "Attack"):
            if attack_id == MARNIE_GRIMMSNARL_ATK1:
                # Main win condition — 136x in match, Marnie's Grimmsnarl ex burst
                score = 10000.0
                if close_game:
                    score += 3000.0
            elif attack_id == GRIMMSNARL_ATK2:
                # Secondary strong attack (37x)
                score = 8000.0
            elif attack_id == GRIMMSNARL_ATK1:
                # Attack 1 (22x)
                score = 7000.0
            elif attack_id in (GRIMMSNARL_ATK3, GRIMMSNARL_ATK4):
                score = 5000.0
            else:
                score = 2000.0

        # ─── PRIORITY 2: ABILITY ────────────────────────────────────────────────
        elif opt_type in (17, "Ability"):
            # Munkidori Adrenaline Brains: distribute damage counters strategically
            score = 9500.0  # Almost always want this

        # ─── PRIORITY 3: EVOLVE ────────────────────────────────────────────────
        elif opt_type in (9, 12, "Evolve"):
            if card_id == MARNIE_GRIMMSNARL_EX_ID:
                score = 9000.0  # Evolve to Marnie's Grimmsnarl ex (main attacker)
            elif card_id == GRIMMSNARL_EX_ID:
                score = 8500.0  # Evolve to Grimmsnarl ex (backup)
            elif card_id == MORGREM_ID:
                score = 8000.0  # Evolve Impidimp → Morgrem
            elif card_id == MORPEKO_EVOLVE_ID:
                score = 7500.0

        # ─── PRIORITY 4: TRAINERS / SUPPORTERS ─────────────────────────────────
        elif opt_type in (7, "Play"):
            if card_id == SPIKEMUTH_GYM_ID:
                # Spikemuth Gym: boost Darkness attacks / discard opponent's stadium
                score = 8500.0

            elif card_id == POFFIN_ID:
                # Poffin: bench 2 basics (Impidimp + Munkidori / Morpeko)
                if len(bench_ids) < 3:
                    score = 8000.0
                elif not bench_full:
                    score = 5500.0
                else:
                    score = 1500.0

            elif card_id == MUNKIDORI_ID:
                # Munkidori: Adrenaline Brains = core damage routing
                if not has_munkidori and not bench_full:
                    score = 7500.0
                else:
                    score = 1500.0

            elif card_id == IMPIDIMP_ID:
                # Deploy Impidimp for Grimmsnarl evolution chain
                if grimsnarl_ct < 2 and not bench_full:
                    score = 7000.0
                else:
                    score = 2000.0

            elif card_id == MARNIE_MORPEKO_ID:
                # Marnie's Morpeko: speed support
                if not bench_full:
                    score = 6500.0
                else:
                    score = 2000.0

            elif card_id == PETREL_ID:
                # Team Rocket's Petrel: recycle cards from discard
                score = 6000.0 if my_prizes <= 4 else 4000.0

            elif card_id in (UNFAIR_STAMP_ID, UNFAIR_STAMP_ALT):
                # Unfair Stamp: collapse opponent's hand after a KO
                if opp_prizes < 6:
                    score = 7000.0
                else:
                    score = 3000.0

            elif card_id in (UNKNOWN_SUPP_1231, UNKNOWN_SUPP_1225):
                # Unknown supporters — treat as draw/search
                score = 5000.0 if ctx["my_hand_len"] < 4 else 3000.0

            elif card_id == FEZANDIPITI_EX_ID:
                score = 5000.0 if not bench_full else 2000.0

            elif card_id == POKE_PAD_ID:
                score = 4500.0

            elif card_id == MISC_305:
                score = 3000.0

        # ─── PRIORITY 5: ENERGY ATTACHMENT ─────────────────────────────────────
        elif opt_type in (8, "Attach"):
            if active_id == MARNIE_GRIMMSNARL_EX_ID:
                score = 7500.0
            elif MARNIE_GRIMMSNARL_EX_ID in bench_ids:
                score = 7000.0
            elif active_id == GRIMMSNARL_EX_ID or GRIMMSNARL_EX_ID in bench_ids:
                score = 6000.0
            elif active_id == MUNKIDORI_ID or MUNKIDORI_ID in bench_ids:
                score = 4000.0  # Munkidori needs Psychic energy for ability
            else:
                score = 3000.0

        # ─── PRIORITY 6: RETREAT ─────────────────────────────────────────────
        elif opt_type in (5, "Retreat"):
            if active_id == MARNIE_GRIMMSNARL_EX_ID and active_en >= 2 and active_hp > 80:
                score = -9000.0  # Never retreat powered Marnie's Grimmsnarl ex
            elif active_id == IMPIDIMP_ID and active_hp < 40:
                score = 3000.0  # Retreat damaged Impidimp
            else:
                score = 500.0

        if score > best_score:
            best_score = score
            best_idx   = idx

    return [best_idx]
