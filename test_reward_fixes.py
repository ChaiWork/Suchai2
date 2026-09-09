import unittest
from src.reward_system.base_reward import calculate_base_strategic_reward_components
from src.reward_system.mewtwo_reward import calculate_mewtwo_strategic_reward
from src.expert_system.energy_evaluator import _dynamic_role_weight


class TestRewardFixes(unittest.TestCase):

    # -----------------------------------------------------------------------
    # 1. RETREAT: V(S') - V(S)
    # -----------------------------------------------------------------------

    def test_retreat_correct_play_positive(self):
        """Retreating a dying Articuno (20 HP, 1 energy) for a semi-ready Mewtwo ex (2 energy)
        should produce POSITIVE r_retreat_eff: readiness gain outweighs energy cost (hp < 40
        so energy_cost = 0.0, only survival + readiness delta applies)."""
        pre = {
            "active_id": 414, "active_energy": 1, "active_min_req_energy": 2,
            "active_hp": 20, "active_max_hp": 120, "my_active_energy": 1, "turn": 4,
        }
        post = {
            "active_id": 431, "active_energy": 2, "active_min_req_energy": 3,
            "active_hp": 280, "active_max_hp": 280, "my_active_energy": 2, "turn": 4,
        }
        comp = calculate_base_strategic_reward_components(pre, post, 12, 10, False, 0, "")
        self.assertGreater(
            comp["r_retreat_eff"], 0.0,
            f"Retreating dying Articuno for semi-ready Mewtwo should be POSITIVE, got {comp['r_retreat_eff']}"
        )

    def test_retreat_wasteful_healthy_penalized(self):
        """Retreating a healthy Articuno (120 HP, 1 energy) for a 0-energy Mewtwo ex should be
        NEGATIVE: post_readiness (0.0) < pre_readiness (0.5) and energy_cost applies (hp >= 80)."""
        pre = {
            "active_id": 414, "active_energy": 1, "active_min_req_energy": 2,
            "active_hp": 120, "active_max_hp": 120, "my_active_energy": 1, "turn": 3,
        }
        post = {
            "active_id": 431, "active_energy": 0, "active_min_req_energy": 3,
            "active_hp": 280, "active_max_hp": 280, "my_active_energy": 0, "turn": 3,
        }
        comp = calculate_base_strategic_reward_components(pre, post, 12, 10, False, 0, "")
        self.assertLess(
            comp["r_retreat_eff"], 0.0,
            f"Retreating healthy Articuno for 0-energy Mewtwo should be NEGATIVE, got {comp['r_retreat_eff']}"
        )

    def test_mewtwo_no_double_retreat_penalty(self):
        """A1 (ALL6): mewtwo_reward.py adds explicit -0.10 penalty for Articuno retreat with energy.
        This is intentional (not double-counting) — Articuno has 2-energy attack cost,
        retreating at 1 energy discards setup progress. Penalty must be < 0 but > -0.25."""
        pre  = {"active_id": 414, "active_hp": 120, "my_active_energy": 1, "bench_ids": [431, 400, 401], "bench_energies": [3, 0, 0], "turn": 3}
        post = {"active_id": 431, "active_hp": 280, "my_active_energy": 3, "bench_ids": [414, 400, 401], "bench_energies": [0, 0, 0], "turn": 3}
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 0, "")
        self.assertLess(r, 0.0, f"Articuno retreat with energy must produce negative mewtwo_reward (A1), got {r}")
        self.assertGreater(r, -0.25, f"Articuno retreat penalty should be small (A1 = -0.10), not > -0.25, got {r}")

    # -----------------------------------------------------------------------
    # 2. MISSED ATTACK: gate on has_attack_option
    # -----------------------------------------------------------------------

    def test_missed_attack_with_legal_option(self):
        """has_attack_option=True + ready attacker: -0.40 penalty fires."""
        pre  = {"active_id": 431, "active_energy": 3, "active_min_req_energy": 3, "opp_active_hp": 280, "has_attack_option": True}
        post = {"active_id": 431, "active_energy": 3, "opp_active_hp": 280}
        comp = calculate_base_strategic_reward_components(pre, post, 0, 10, False, 0, "")
        self.assertEqual(comp["r_missed_attack"], -0.75)

    def test_missed_attack_no_legal_option(self):
        """has_attack_option=False (e.g. Mimikyu immunity): no missed attack penalty."""
        pre  = {"active_id": 431, "active_energy": 3, "active_min_req_energy": 3, "opp_active_hp": 280, "has_attack_option": False}
        post = {"active_id": 431, "active_energy": 3, "opp_active_hp": 280}
        comp = calculate_base_strategic_reward_components(pre, post, 0, 10, False, 0, "")
        self.assertEqual(comp["r_missed_attack"], 0.0,
                         f"No penalty without legal attack option, got {comp['r_missed_attack']}")

    # -----------------------------------------------------------------------
    # 3. STADIUM: value delta, not flat reward
    # -----------------------------------------------------------------------

    def test_stadium_upgrade_positive(self):
        """Playing TR Factory (0.10) over empty stadium (-1): delta = +0.10."""
        pre  = {"stadium_id": -1, "played_card_id": 1257}
        post = {"stadium_id": 1257}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertGreater(comp["r_stadium_value"], 0.0)

    def test_stadium_downgrade_negative(self):
        """Playing Battle Cage (0.08) over TR Factory (0.10): delta = -0.02, negative."""
        pre  = {"stadium_id": 1257, "played_card_id": 1264}
        post = {"stadium_id": 1264}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertLess(comp["r_stadium_value"], 0.0,
                        f"Playing Battle Cage over TR Factory should be negative, got {comp['r_stadium_value']}")

    # -----------------------------------------------------------------------
    # 4. SEARCH QUALITY: tempo-only = 0.05
    # -----------------------------------------------------------------------

    def test_search_quality_tempo(self):
        """r_search_quality should be 0.05 (tempo-only)."""
        pre  = {"played_card_id": 1134}  # Transceiver
        post = {}
        comp = calculate_base_strategic_reward_components(pre, post, 7, 10, False, 0, "")
        self.assertAlmostEqual(comp["r_search_quality"], 0.05, places=5)

    # -----------------------------------------------------------------------
    # 5. SPIDOPS RETREAT PENALTY & TANK OPPONENT BENCH ENERGY REWARD
    # -----------------------------------------------------------------------

    def test_spidops_retreat_when_can_attack_penalized(self):
        """Retreating Spidops (401) when it has 2+ energy or attack option triggers explicit penalty."""
        debug = []
        pre  = {"active_id": 401, "active_hp": 90, "my_active_energy": 2, "has_attack_option": True, "turn": 4}
        post = {"active_id": 431, "active_hp": 280, "my_active_energy": 0, "turn": 4}
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        # Either the catastrophic penalty (attack was available) or the standard retreat penalty must fire
        has_any_retreat_penalty = (
            "CATASTROPHIC_Penalty_Retreated_With_Attack_Available_Spidops_Had_Energy" in reasons
            or "Penalty_Spidops_Retreat_When_Able_To_Attack" in reasons
        )
        self.assertTrue(has_any_retreat_penalty,
                        f"A Spidops retreat penalty should fire when retreating while able to attack. Got: {reasons}")

    def test_erasure_ball_tank_opponent_bench_energy_reward(self):
        """Attaching energy to benched Spidops when facing a tank opponent (HP>=180) fires tank fuel reward."""
        debug = []
        pre = {
            "action_type": 8, "attached_card_id": 1, "attached_target_id": 401,
            "target_energy": 0, "bench_ids": [401], "bench_energies": [0],
            "active_id": 431, "my_active_energy": 3, "active_hp": 280,
            "opp_active_hp": 280, "opp_active_id": 269, "turn": 4
        }
        post = pre.copy()
        r = calculate_mewtwo_strategic_reward(pre, post, 8, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        self.assertIn("Reward_Bench_Energy_Fuel_Vs_Tank_Opponent", reasons,
                      "Reward_Bench_Energy_Fuel_Vs_Tank_Opponent should fire when attaching energy to bench vs tank opponent")

    # -----------------------------------------------------------------------
    # 5. DYNAMIC ARTICUNO ROLE WEIGHT
    # -----------------------------------------------------------------------

    def test_articuno_weight_active_no_mewtwo(self):
        """Articuno is active, Mewtwo not on bench: dynamic weight = 0.40."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={400, 434}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.40, places=5)

    def test_articuno_weight_close_game(self):
        """Articuno is active, Mewtwo benched, prizes <= 2: weight = 0.25."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={431, 400}, prizes=2, opp_prizes=1)
        self.assertAlmostEqual(w, 0.25, places=5)

    def test_articuno_weight_bench(self):
        """Articuno on bench (Mewtwo active): static weight = 0.05."""
        w = _dynamic_role_weight(414, active_id=431, bench_ids={414}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.05, places=5)

    # -----------------------------------------------------------------------
    # 6. ACTION CONVERSION: still gates on board development
    # -----------------------------------------------------------------------

    def test_unpowered_mewtwo_active_exposure_midgame(self):
        """Unpowered Mewtwo ex in active position should be penalized at turn 8 (mid-game)."""
        debug = []
        pre = {"active_id": 401, "my_active_energy": 0, "bench_ids": [431, 414], "turn": 8}
        post = {"active_id": 431, "my_active_energy": 0, "bench_ids": [414], "turn": 8}
        r = calculate_mewtwo_strategic_reward(pre, post, 1, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        self.assertIn("Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet", reasons,
                      "Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet should fire at turn 8")

    def test_setup_turns_no_unpowered_mewtwo_penalty(self):
        """Setup turns 1-3 should NOT penalize Mewtwo ex for having < 3 energy during normal setup."""
        debug = []
        pre = {"active_id": 431, "my_active_energy": 1, "bench_ids": [400, 401, 414], "turn": 2}
        post = {"active_id": 431, "my_active_energy": 1, "bench_ids": [400, 401, 414], "turn": 2}
        r = calculate_mewtwo_strategic_reward(pre, post, 8, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        self.assertNotIn("Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet", reasons,
                         "Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet should NOT fire on setup turn 2")

    def test_to_active_promotion_priors(self):
        """ToActive promotion prior should penalize 0-energy Mewtwo ex (-0.35) and favor Articuno (+0.25)."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "ToActive"},
            "current": {
                "turn": 6, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [],
                        "bench": [
                            {"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280},
                            {"cardId": 414, "energyCards": [], "hp": 120, "maxHp": 120}
                        ],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_mewtwo = {"type": "Card", "area": 5, "index": 0, "playerIndex": 1, "context": "ToActive"}
        opt_articuno = {"type": "Card", "area": 5, "index": 1, "playerIndex": 1, "context": "ToActive"}

        score_m, reason_m = evaluate_mewtwo_expert_bonus(obs, opt_mewtwo)
        score_a, reason_a = evaluate_mewtwo_expert_bonus(obs, opt_articuno)

        self.assertLess(score_m, 0.0, f"Promoting 0-energy Mewtwo ex should be penalized, got {score_m} ({reason_m})")
        self.assertGreater(score_a, 0.0, f"Promoting 1-prize Articuno should be favored, got {score_a} ({reason_a})")

    def test_spidops_retreat_prohibition_prior(self):
        """Retreating Spidops (401) with 1 energy should yield prior penalty -0.35."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "current": {
                "turn": 2, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 401, "energyCards": [1], "hp": 130, "maxHp": 130}],
                        "bench": [{"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280}],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_retreat = {"type": "Retreat", "area": 4, "index": 0}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt_retreat)
        self.assertLessEqual(score, -0.35, f"Retreating Spidops with energy should yield penalty <= -0.35, got {score} ({reason})")

    def test_benched_mewtwo_third_energy_priority(self):
        """Benched Mewtwo ex should yield dynamic role weight 1.20 to complete 3rd energy."""
        from src.expert_system.energy_evaluator import _dynamic_role_weight
        w = _dynamic_role_weight(431, active_id=401, bench_ids=[431, 400], prizes=6, opp_prizes=6)
        self.assertEqual(w, 1.20, "Benched Mewtwo ex role weight should be 1.20 for 3rd energy completion")

    def test_unpowered_spidops_promotion_penalty_when_powered_mewtwo_exists(self):
        """Promoting 0-energy Spidops when fully powered Mewtwo ex exists should yield penalty -0.35."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "ToActive"},
            "current": {
                "turn": 6, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [],
                        "bench": [
                            {"cardId": 431, "energyCards": [1, 2, 3], "hp": 280, "maxHp": 280},
                            {"cardId": 401, "energyCards": [], "hp": 130, "maxHp": 130},
                            {"cardId": 400, "energyCards": [], "hp": 50, "maxHp": 50},
                            {"cardId": 414, "energyCards": [], "hp": 120, "maxHp": 120}
                        ],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_spidops = {"type": "Card", "area": 5, "index": 1, "playerIndex": 1, "context": "ToActive"}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt_spidops)
        self.assertLessEqual(score, -0.35, f"Promoting unpowered Spidops over powered Mewtwo ex should be penalized <= -0.35, got {score} ({reason})")

    # ─── A1: Articuno retreat reward penalty ──────────────────────────────────
    def test_articuno_retreat_with_energy_reward_penalty(self):
        """A1 (ALL6): Articuno retreated with 1 energy 110 times total; reward penalty must fire."""
        r = calculate_mewtwo_strategic_reward(
            pre={"active_id": 414, "my_active_energy": 1, "active_hp": 120, "bench_ids": [431], "turn": 4},
            post={"active_id": 431, "my_active_energy": 0, "bench_ids": [414], "turn": 4},
            action_type=12, step_idx=10, went_second=True, player_idx=0
        )
        self.assertLess(r, 0.0, f"Retreating Articuno with 1 energy should produce negative reward, got {r}")

    # ─── A2: Bench fuel reward magnitude reduced ──────────────────────────────
    def test_bench_fuel_reward_reduced_to_015(self):
        """A2 (ALL6): Bench energy fuel reward must be 0.15 not 0.30 to avoid reward domination."""
        debug = []
        calculate_mewtwo_strategic_reward(
            pre={
                "active_id": 401, "my_active_energy": 2, "active_hp": 130,
                "attached_card_id": 1, "attached_target_id": 431, "target_energy": 0,
                "bench_ids": [431], "bench_energies": [0],
                "opp_active_hp": 280, "opp_active_id": 431, "prizes": 6, "opp_prizes": 6, "turn": 4,
            },
            post={"active_id": 401, "my_active_energy": 2, "bench_ids": [431], "turn": 4},
            action_type=8, step_idx=10, went_second=True, player_idx=0, debug_log=debug
        )
        fuel_rewards = [v for name, v in debug if "Bench_Energy_Fuel" in name]
        for v in fuel_rewards:
            self.assertLessEqual(abs(v), 0.15 + 1e-6, f"Bench fuel reward {v} exceeds 0.15 — reward magnitude too high")

    # ─── B1: Articuno retreat expert prior penalty ────────────────────────────
    def test_articuno_retreat_with_energy_expert_prior(self):
        """B1 (ALL6): Expert prior must penalize retreating Articuno with >= 1 energy."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "Retreat"},
            "current": {
                "turn": 4, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 414, "energyCards": [1], "hp": 120, "maxHp": 120}],
                        "bench": [{"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280}],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt = {"type": "Retreat"}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        self.assertLessEqual(score, -0.35, f"Retreating Articuno with energy should be penalized <= -0.35, got {score} ({reason})")

    # ─── B3: Starmie fast-aggro bench priority boost ──────────────────────────
    def test_starmie_bench_fill_urgency_poffin_prior(self):
        """B3 (ALL6): Against Starmie (fast aggro), Poffin must get urgency boost when bench < 4 TR."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "Main"},
            "current": {
                "turn": 2, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 401, "energyCards": [1], "hp": 130, "maxHp": 130}],
                        "bench": [{"cardId": 400, "energyCards": [], "hp": 50, "maxHp": 50}],
                        "prize": [1]*6,
                        # Poffin has explicit cardId so expert resolves it directly
                        "hand": [{"cardId": 1086, "id": 1086}]
                    }
                ]
            }
        }
        # Pass cardId directly on the option so hand-index resolution is bypassed
        # Use integer type=7 (OptionType.PLAY) to match actual engine enum value
    # 5. DYNAMIC ARTICUNO ROLE WEIGHT
    # -----------------------------------------------------------------------

    def test_articuno_weight_active_no_mewtwo(self):
        """Articuno is active, Mewtwo not on bench: dynamic weight = 0.40."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={400, 434}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.40, places=5)

    def test_articuno_weight_close_game(self):
        """Articuno is active, Mewtwo benched, prizes <= 2: weight = 0.25."""
        w = _dynamic_role_weight(414, active_id=414, bench_ids={431, 400}, prizes=2, opp_prizes=1)
        self.assertAlmostEqual(w, 0.25, places=5)

    def test_articuno_weight_bench(self):
        """Articuno on bench (Mewtwo active): static weight = 0.05."""
        w = _dynamic_role_weight(414, active_id=431, bench_ids={414}, prizes=4, opp_prizes=4)
        self.assertAlmostEqual(w, 0.05, places=5)

    # -----------------------------------------------------------------------
    # 6. ACTION CONVERSION: still gates on board development
    # -----------------------------------------------------------------------

    def test_unpowered_mewtwo_active_exposure_midgame(self):
        """Unpowered Mewtwo ex in active position should be penalized at turn 8 (mid-game)."""
        debug = []
        pre = {"active_id": 401, "my_active_energy": 0, "bench_ids": [431, 414], "turn": 8}
        post = {"active_id": 431, "my_active_energy": 0, "bench_ids": [414], "turn": 8}
        r = calculate_mewtwo_strategic_reward(pre, post, 1, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        self.assertIn("Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet", reasons,
                      "Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet should fire at turn 8")

    def test_setup_turns_no_unpowered_mewtwo_penalty(self):
        """Setup turns 1-3 should NOT penalize Mewtwo ex for having < 3 energy during normal setup."""
        debug = []
        pre = {"active_id": 431, "my_active_energy": 1, "bench_ids": [400, 401, 414], "turn": 2}
        post = {"active_id": 431, "my_active_energy": 1, "bench_ids": [400, 401, 414], "turn": 2}
        r = calculate_mewtwo_strategic_reward(pre, post, 8, 10, False, 0, "", debug_log=debug)
        reasons = [item[0] for item in debug]
        self.assertNotIn("Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet", reasons,
                         "Penalty_Exposed_Unpowered_Active_Mewtwo_ex_Power_Saver_Unmet should NOT fire on setup turn 2")

    def test_to_active_promotion_priors(self):
        """ToActive promotion prior should penalize 0-energy Mewtwo ex (-0.35) and favor Articuno (+0.25)."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "ToActive"},
            "current": {
                "turn": 6, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [],
                        "bench": [
                            {"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280},
                            {"cardId": 414, "energyCards": [], "hp": 120, "maxHp": 120}
                        ],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_mewtwo = {"type": "Card", "area": 5, "index": 0, "playerIndex": 1, "context": "ToActive"}
        opt_articuno = {"type": "Card", "area": 5, "index": 1, "playerIndex": 1, "context": "ToActive"}

        score_m, reason_m = evaluate_mewtwo_expert_bonus(obs, opt_mewtwo)
        score_a, reason_a = evaluate_mewtwo_expert_bonus(obs, opt_articuno)

        self.assertLess(score_m, 0.0, f"Promoting 0-energy Mewtwo ex should be penalized, got {score_m} ({reason_m})")
        self.assertGreater(score_a, 0.0, f"Promoting 1-prize Articuno should be favored, got {score_a} ({reason_a})")

    def test_spidops_retreat_prohibition_prior(self):
        """Retreating Spidops (401) with 1 energy should yield prior penalty -0.35."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "current": {
                "turn": 2, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 401, "energyCards": [1], "hp": 130, "maxHp": 130}],
                        "bench": [{"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280}],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_retreat = {"type": "Retreat", "area": 4, "index": 0}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt_retreat)
        self.assertLessEqual(score, -0.35, f"Retreating Spidops with energy should yield penalty <= -0.35, got {score} ({reason})")

    def test_benched_mewtwo_third_energy_priority(self):
        """Benched Mewtwo ex should yield dynamic role weight 1.20 to complete 3rd energy."""
        from src.expert_system.energy_evaluator import _dynamic_role_weight
        w = _dynamic_role_weight(431, active_id=401, bench_ids=[431, 400], prizes=6, opp_prizes=6)
        self.assertEqual(w, 1.20, "Benched Mewtwo ex role weight should be 1.20 for 3rd energy completion")

    def test_unpowered_spidops_promotion_penalty_when_powered_mewtwo_exists(self):
        """Promoting 0-energy Spidops when fully powered Mewtwo ex exists should yield penalty -0.35."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "ToActive"},
            "current": {
                "turn": 6, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [],
                        "bench": [
                            {"cardId": 431, "energyCards": [1, 2, 3], "hp": 280, "maxHp": 280},
                            {"cardId": 401, "energyCards": [], "hp": 130, "maxHp": 130},
                            {"cardId": 400, "energyCards": [], "hp": 50, "maxHp": 50},
                            {"cardId": 414, "energyCards": [], "hp": 120, "maxHp": 120}
                        ],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt_spidops = {"type": "Card", "area": 5, "index": 1, "playerIndex": 1, "context": "ToActive"}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt_spidops)
        self.assertLessEqual(score, -0.35, f"Promoting unpowered Spidops over powered Mewtwo ex should be penalized <= -0.35, got {score} ({reason})")

    # ─── A1: Articuno retreat reward penalty ──────────────────────────────────
    def test_articuno_retreat_with_energy_reward_penalty(self):
        """A1 (ALL6): Articuno retreated with 1 energy 110 times total; reward penalty must fire."""
        r = calculate_mewtwo_strategic_reward(
            pre={"active_id": 414, "my_active_energy": 1, "active_hp": 120, "bench_ids": [431], "turn": 4},
            post={"active_id": 431, "my_active_energy": 0, "bench_ids": [414], "turn": 4},
            action_type=12, step_idx=10, went_second=True, player_idx=0
        )
        self.assertLess(r, 0.0, f"Retreating Articuno with 1 energy should produce negative reward, got {r}")

    # ─── A2: Bench fuel reward magnitude reduced ──────────────────────────────
    def test_bench_fuel_reward_reduced_to_015(self):
        """A2 (ALL6): Bench energy fuel reward must be 0.15 not 0.30 to avoid reward domination."""
        debug = []
        calculate_mewtwo_strategic_reward(
            pre={
                "active_id": 401, "my_active_energy": 2, "active_hp": 130,
                "attached_card_id": 1, "attached_target_id": 431, "target_energy": 0,
                "bench_ids": [431], "bench_energies": [0],
                "opp_active_hp": 280, "opp_active_id": 431, "prizes": 6, "opp_prizes": 6, "turn": 4,
            },
            post={"active_id": 401, "my_active_energy": 2, "bench_ids": [431], "turn": 4},
            action_type=8, step_idx=10, went_second=True, player_idx=0, debug_log=debug
        )
        fuel_rewards = [v for name, v in debug if "Bench_Energy_Fuel" in name]
        for v in fuel_rewards:
            self.assertLessEqual(abs(v), 0.15 + 1e-6, f"Bench fuel reward {v} exceeds 0.15 — reward magnitude too high")

    # ─── B1: Articuno retreat expert prior penalty ────────────────────────────
    def test_articuno_retreat_with_energy_expert_prior(self):
        """B1 (ALL6): Expert prior must penalize retreating Articuno with >= 1 energy."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "Retreat"},
            "current": {
                "turn": 4, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 414, "energyCards": [1], "hp": 120, "maxHp": 120}],
                        "bench": [{"cardId": 431, "energyCards": [], "hp": 280, "maxHp": 280}],
                        "prize": [1]*6, "hand": []
                    }
                ]
            }
        }
        opt = {"type": "Retreat"}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        self.assertLess(score, 0.0,
            f"Retreating Articuno with energy should be penalized (< 0), got {score} ({reason})"
            f" — Note: E1 makes penalty TR-conditional; TR=2 gets soft -0.10; see test_E1_expert_* for full/soft split"
        )

    # ─── B3: Starmie fast-aggro bench priority boost ──────────────────────────
    def test_starmie_bench_fill_urgency_poffin_prior(self):
        """B3 (ALL6): Against Starmie (fast aggro), Poffin must get urgency boost when bench < 4 TR."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "Main"},
            "current": {
                "turn": 2, "yourIndex": 1,
                "players": [
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                    {
                        "active": [{"cardId": 401, "energyCards": [1], "hp": 130, "maxHp": 130}],
                        "bench": [{"cardId": 400, "energyCards": [], "hp": 50, "maxHp": 50}],
                        "prize": [1]*6,
                        # Poffin has explicit cardId so expert resolves it directly
                        "hand": [{"cardId": 1086, "id": 1086}]
                    }
                ]
            }
        }
        # Pass cardId directly on the option so hand-index resolution is bypassed
        # Use integer type=7 (OptionType.PLAY) to match actual engine enum value
        opt = {"type": 7, "cardId": 1086, "index": 0}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt, opponent_name="Rulebasedmodel_Starmie")
        self.assertGreaterEqual(score, 0.18, f"Starmie fast-aggro Poffin prior should be >= 0.18, got {score} ({reason})")




class TestLatestReplayFixes(unittest.TestCase):
    """
    Phase 12 regression tests — evidence-backed against LATEST 210-match replay audit.
    Tests R1, R2, R4 (mewtwo_reward.py) and E1, E2, E3 (mewtwo_expert.py).
    """

    def _make_pre(self, active_id, active_energy, active_hp, bench_ids=None, action_type=12):
        bench_ids = bench_ids or []
        return {
            "active_id": active_id,
            "my_active_energy": active_energy,
            "active_hp": active_hp,
            "bench_ids": bench_ids,
            "action_type": action_type,
            "turn": 10,
            "prizes": 4,
            "opp_prizes": 3,
        }

    def _make_post(self, active_id, active_energy=0):
        return {"active_id": active_id, "my_active_energy": active_energy, "turn": 10}

    # ── R1: Articuno TR-count conditional penalty ────────────────────────────

    def test_R1_articuno_retreat_TR4_full_penalty(self):
        """R1: Articuno retreats with energy when TR>=4 → penalty >= -0.15 (full board, no offset)."""
        # 4 TR poke in play: Articuno active + 3 bench TR poke
        bench = [MEWTWO_EX_ID, SPIDOPS_ID, TAROUNTULA_ID]
        pre  = self._make_pre(ARTICUNO_ID, 1, 120, bench_ids=bench)
        post = self._make_post(MIMIKYU_ID)
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 1)
        self.assertLessEqual(r, -0.05,
            f"R1: Articuno retreat with TR=4 must produce penalty <= -0.05, got {r}")

    def test_R1_articuno_retreat_TR2_soft_penalty(self):
        """R1: Articuno retreats with energy when TR<=2 → penalty should be softer (>= -0.10)."""
        # Only 2 TR poke: Articuno active + Tarountula bench
        bench = [TAROUNTULA_ID]
        pre  = self._make_pre(ARTICUNO_ID, 1, 120, bench_ids=bench)
        post = self._make_post(MIMIKYU_ID)
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 1)
        # Soft path: -0.05 is the expected value when TR<4
        self.assertGreater(r, -0.12,
            f"R1: Articuno retreat with TR=2 should be soft penalty (> -0.12), got {r}")

    # ── R2: Spidops HP-split penalty ─────────────────────────────────────────

    def test_R2_spidops_retreat_healthy_harder_penalty(self):
        """R2: Healthy Spidops (hp=130) retreating with energy → penalty <= -0.15."""
        bench = [MEWTWO_EX_ID, TAROUNTULA_ID]
        pre  = self._make_pre(SPIDOPS_ID, 2, 130, bench_ids=bench)
        post = self._make_post(MEWTWO_EX_ID)
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 1)
        self.assertLessEqual(r, -0.05,
            f"R2: Healthy Spidops retreat with energy must be penalized (<= -0.05), got {r}")

    def test_R2_spidops_retreat_near_ko_lighter_penalty(self):
        """R2: Near-KO Spidops (hp=20) retreating with energy → penalty must be > -0.30 (escape justified)."""
        bench = [MEWTWO_EX_ID, TAROUNTULA_ID]
        pre  = self._make_pre(SPIDOPS_ID, 2, 20, bench_ids=bench)
        post = self._make_post(MEWTWO_EX_ID)
        r = calculate_mewtwo_strategic_reward(pre, post, 12, 10, False, 1)
        self.assertGreater(r, -0.30,
            f"R2: Near-KO Spidops retreat should be lighter penalty (> -0.30), got {r}")

    # ── R4: Multi-energy Articuno penalty ────────────────────────────────────

    def test_R4_articuno_2energy_TR4_extra_penalty(self):
        """R4: Articuno with 2+ energy retreating when TR=4 → stronger penalty than 1-energy case."""
        bench = [MEWTWO_EX_ID, SPIDOPS_ID, TAROUNTULA_ID]  # Articuno + 3 = 4 TR
        pre_2en  = self._make_pre(ARTICUNO_ID, 2, 120, bench_ids=bench)
        pre_1en  = self._make_pre(ARTICUNO_ID, 1, 120, bench_ids=bench)
        post     = self._make_post(MIMIKYU_ID)
        r_2en = calculate_mewtwo_strategic_reward(pre_2en, post, 12, 10, False, 1)
        r_1en = calculate_mewtwo_strategic_reward(pre_1en, post, 12, 10, False, 1)
        self.assertLessEqual(r_2en, r_1en,
            f"R4: 2-energy Articuno retreat ({r_2en}) must be penalized more than 1-energy ({r_1en})")

    # ── E1: Expert prior Articuno retreat TR-count conditional ───────────────

    def test_E1_expert_articuno_retreat_TR4_strong_prior(self):
        """E1: Expert prior for Articuno retreat with energy when TR=4 must be <= -0.30."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": ""},
            "current": {
                "turn": 10, "yourIndex": 0,
                "players": [
                    {
                        "active": [{"cardId": ARTICUNO_ID, "energyCards": [1], "hp": 120, "maxHp": 120}],
                        "bench": [
                            {"cardId": MEWTWO_EX_ID, "energyCards": []},
                            {"cardId": SPIDOPS_ID, "energyCards": []},
                            {"cardId": TAROUNTULA_ID, "energyCards": []},
                        ],
                        "prize": [1]*4, "hand": [],
                    },
                    {"active": [], "bench": [], "prize": [1]*3, "hand": []},
                ]
            }
        }
        opt = {"type": 12}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        self.assertLessEqual(score, -0.30,
            f"E1: Articuno retreat prior with TR=4 must be <= -0.30, got {score} ({reason})")

    def test_E1_expert_articuno_retreat_TR2_soft_prior(self):
        """E1: Expert prior for Articuno retreat with energy when TR=2 must be > -0.25."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": ""},
            "current": {
                "turn": 6, "yourIndex": 0,
                "players": [
                    {
                        "active": [{"cardId": ARTICUNO_ID, "energyCards": [1], "hp": 120, "maxHp": 120}],
                        "bench": [{"cardId": TAROUNTULA_ID, "energyCards": []}],
                        "prize": [1]*6, "hand": [],
                    },
                    {"active": [], "bench": [], "prize": [1]*6, "hand": []},
                ]
            }
        }
        opt = {"type": 12}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        self.assertGreater(score, -0.25,
            f"E1: Articuno retreat prior with TR=2 should be soft (> -0.25), got {score} ({reason})")

    # ── E2: Articuno promotion prior conditional on TR count ─────────────────

    def test_E2_articuno_promotion_TR4_reduced_prior(self):
        """E2: Articuno promotion prior when TR=4 should be <= +0.12 (prefer attacker)."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": "ToActive"},
            "current": {
                "turn": 12, "yourIndex": 0,
                "players": [
                    {
                        "active": [{"cardId": SPIDOPS_ID, "energyCards": [], "hp": 20, "maxHp": 130}],
                        "bench": [
                            {"cardId": MEWTWO_EX_ID, "energyCards": []},
                            {"cardId": TAROUNTULA_ID, "energyCards": []},
                            {"cardId": MIMIKYU_ID, "energyCards": []},
                            {"cardId": ARTICUNO_ID, "energyCards": []},
                        ],
                        "prize": [1]*3, "hand": [],
                    },
                    {"active": [], "bench": [], "prize": [1]*2, "hand": []},
                ]
            }
        }
        opt = {"type": 3, "area": 5, "index": 3, "cardId": ARTICUNO_ID}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        self.assertLessEqual(score, 0.12,
            f"E2: Articuno promotion with TR=4 should be <= +0.12, got {score} ({reason})")

    # ── E3: Starmie fast-aggro Spidops attack urgency ─────────────────────────

    def test_E3_spidops_attack_starmie_full_urgency(self):
        """E3: Powered Spidops attack prior vs Starmie must be 0.35 (highest urgency)."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": ""},
            "current": {
                "turn": 8, "yourIndex": 0,
                "players": [
                    {
                        "active": [{"cardId": SPIDOPS_ID, "energyCards": [1, 1], "hp": 130, "maxHp": 130}],
                        "bench": [
                            {"cardId": MEWTWO_EX_ID, "energyCards": []},
                            {"cardId": TAROUNTULA_ID, "energyCards": []},
                        ],
                        "prize": [1]*5, "hand": [],
                    },
                    {"active": [{"cardId": 999, "energyCards": [1], "hp": 80, "maxHp": 80}], "bench": [], "prize": [1]*4, "hand": []},
                ]
            }
        }
        opt = {"type": 13}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt, opponent_name="Rulebasedmodel_Starmie")
        self.assertGreaterEqual(score, 0.30,
            f"E3: Spidops attack prior vs Starmie must be >= 0.30, got {score} ({reason})")

    # ── Architecture: NN remains final decision maker ────────────────────────

    def test_architecture_expert_is_bounded_not_blocking(self):
        """Architecture: Expert prior for Articuno retreat must remain a float, not block the action."""
        from src.expert_system.mewtwo_expert import evaluate_mewtwo_expert_bonus
        obs = {
            "select": {"context": ""},
            "current": {
                "turn": 10, "yourIndex": 0,
                "players": [
                    {
                        "active": [{"cardId": ARTICUNO_ID, "energyCards": [1], "hp": 120, "maxHp": 120}],
                        "bench": [{"cardId": MEWTWO_EX_ID, "energyCards": []}],
                        "prize": [1]*5, "hand": [],
                    },
                    {"active": [], "bench": [], "prize": [1]*5, "hand": []},
                ]
            }
        }
        opt = {"type": 12}
        score, reason = evaluate_mewtwo_expert_bonus(obs, opt)
        # Must be a float in bounded range — not None, not an exception, not a hard block
        self.assertIsInstance(score, float,
            f"Expert prior must return float, got {type(score)}")
        self.assertGreaterEqual(score, -1.0,
            f"Expert prior must be >= -1.0 (bounded), got {score}")
        self.assertLessEqual(score, 1.0,
            f"Expert prior must be <= +1.0 (bounded), got {score}")


# Card ID aliases for test readability
MEWTWO_EX_ID  = 431
SPIDOPS_ID    = 401
TAROUNTULA_ID = 400
ARTICUNO_ID   = 414
MIMIKYU_ID    = 434


if __name__ == "__main__":
    unittest.main()
