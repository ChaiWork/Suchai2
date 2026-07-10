#!/usr/bin/env python3
"""
Learn From Mistake — PTCG Battle Replay Analyzer
=================================================
Reads a battle_viewer JSON replay and generates a Markdown post-mortem
report summarizing what went wrong, so RL reward shaping can be improved.

Usage:
    python Learn_from_mistake/analyze_loss.py <replay.json> [output.md]

If output.md is omitted, saves to Learn_from_mistake/<episode_id>_report.md
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# ── Helpers ────────────────────────────────────────────────────────────────

def load_replay(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_card_names(data: dict) -> dict:
    """Build a card_id -> name lookup from all state data in the replay."""
    names = {}
    for step_list in data.get("steps", []):
        for agent_step in step_list:
            for vis in agent_step.get("visualize", []):
                curr = vis.get("current", {})
                if not curr:
                    continue
                for player in curr.get("players", []):
                    for zone in ["deck", "hand", "discard", "active", "bench", "prize"]:
                        for c in player.get(zone, []):
                            if isinstance(c, dict) and "id" in c and "name" in c:
                                names[c["id"]] = c["name"]
    return names


def cname(card_id: int, names: dict) -> str:
    return names.get(card_id, f"card#{card_id}")


# ── Extraction ─────────────────────────────────────────────────────────────

def extract_events(data: dict, names: dict) -> list:
    """Extract a flat list of key game events with turn numbers."""
    events = []
    seen = set()

    for step_list in data.get("steps", []):
        for agent_step in step_list:
            for vis in agent_step.get("visualize", []):
                curr = vis.get("current", {})
                turn = curr.get("turn", 0) if curr else 0
                energy_flag = curr.get("energyAttached", False) if curr else False

                for log in vis.get("logs", []):
                    key = json.dumps(log, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)

                    t = log.get("type", "")
                    p = log.get("playerIndex", -1)

                    if t == "Attack":
                        events.append({
                            "turn": turn, "type": "Attack", "player": p,
                            "card": cname(log.get("cardId", 0), names),
                            "attack_id": log.get("attackId"),
                        })
                    elif t == "HpChange":
                        events.append({
                            "turn": turn, "type": "HpChange", "player": p,
                            "card": cname(log.get("cardId", 0), names),
                            "value": log.get("value", 0),
                        })
                    elif t == "Evolve":
                        events.append({
                            "turn": turn, "type": "Evolve", "player": p,
                            "from": cname(log.get("cardIdTarget", 0), names),
                            "to": cname(log.get("cardId", 0), names),
                        })
                    elif t == "Attach":
                        events.append({
                            "turn": turn, "type": "Attach", "player": p,
                            "energy": cname(log.get("cardId", 0), names),
                            "target": cname(log.get("cardIdTarget", 0), names),
                        })
                    elif t == "Play":
                        events.append({
                            "turn": turn, "type": "Play", "player": p,
                            "card": cname(log.get("cardId", 0), names),
                        })
                    elif t == "Switch":
                        events.append({
                            "turn": turn, "type": "Switch", "player": p,
                            "from": cname(log.get("cardIdActive", 0), names),
                            "to": cname(log.get("cardIdBench", 0), names),
                        })
                    elif t == "TurnEnd":
                        events.append({
                            "turn": turn, "type": "TurnEnd", "player": p,
                            "energy_attached": energy_flag,
                        })
                    elif t == "Result":
                        events.append({
                            "turn": turn, "type": "Result",
                            "result": log.get("result"),
                            "reason": log.get("reason"),
                        })

    return events


def extract_prize_progression(data: dict) -> list:
    """Track prize count changes over time."""
    progression = []
    last = [None, None]

    for step_list in data.get("steps", []):
        for agent_step in step_list:
            for vis in agent_step.get("visualize", []):
                curr = vis.get("current", {})
                if not curr:
                    continue
                players = curr.get("players", [])
                turn = curr.get("turn", 0)
                if len(players) < 2:
                    continue
                p0 = len(players[0].get("prize", []))
                p1 = len(players[1].get("prize", []))
                if [p0, p1] != last:
                    if last[0] is not None:
                        progression.append({
                            "turn": turn,
                            "p0_prizes": p0,
                            "p1_prizes": p1,
                            "p0_took": max(0, last[0] - p0),
                            "p1_took": max(0, last[1] - p1),
                        })
                    last = [p0, p1]

    return progression


def extract_turn_snapshots(data: dict, names: dict) -> list:
    """Extract per-turn board snapshots for mistake detection."""
    snapshots = {}
    prev_key = None

    for step_list in data.get("steps", []):
        for agent_step in step_list:
            for vis in agent_step.get("visualize", []):
                curr = vis.get("current", {})
                if not curr:
                    continue
                players = curr.get("players", [])
                turn = curr.get("turn", 0)
                if len(players) < 2:
                    continue

                def active_info(p):
                    a = p.get("active", [])
                    if not a:
                        return None
                    card = a[0]
                    return {
                        "name": card.get("name", "?"),
                        "hp": card.get("hp"),
                        "max_hp": card.get("maxHp"),
                        "energy_count": len(card.get("energyCards", [])),
                    }

                snap = {
                    "turn": turn,
                    "energy_attached": curr.get("energyAttached", False),
                    "p0": {
                        "active": active_info(players[0]),
                        "bench_count": len([b for b in players[0].get("bench", []) if b]),
                        "bench_names": [b.get("name", "?") for b in players[0].get("bench", []) if b],
                        "prizes": len(players[0].get("prize", [])),
                        "hand_count": players[0].get("handCount", 0),
                        "deck_count": players[0].get("deckCount", 0),
                    },
                    "p1": {
                        "active": active_info(players[1]),
                        "bench_count": len([b for b in players[1].get("bench", []) if b]),
                        "bench_names": [b.get("name", "?") for b in players[1].get("bench", []) if b],
                        "prizes": len(players[1].get("prize", [])),
                    },
                }
                key = (turn, snap["p0"]["bench_count"], snap["p0"]["prizes"], snap["p1"]["prizes"])
                if key != prev_key:
                    snapshots[turn] = snap
                    prev_key = key

    return list(snapshots.values())


# ── Mistake Detection ──────────────────────────────────────────────────────

RESULT_REASONS = {0: "All prizes taken", 1: "Deck out", 2: "No basic Pokemon in hand", 3: "All Pokemon KO'd"}


def detect_mistakes(events: list, snapshots: list, result_player: int) -> list:
    """Detect strategic mistakes made by the losing agent (P0 = Suchai)."""
    mistakes = []
    p0_attacked_turns = set()
    p0_energy_turns = set()
    p0_play_turns = defaultdict(list)

    for e in events:
        if e["type"] == "Attack" and e["player"] == 0:
            p0_attacked_turns.add(e["turn"])
        if e["type"] == "Attach" and e["player"] == 0:
            p0_energy_turns.add(e["turn"])
        if e["type"] == "Play" and e["player"] == 0:
            p0_play_turns[e["turn"]].append(e["card"])

    for snap in snapshots:
        t = snap["turn"]
        if t == 0:
            continue

        p0 = snap["p0"]
        p1 = snap["p1"]
        active = p0["active"]

        # Mistake: Empty bench danger
        if p0["bench_count"] == 0 and active:
            mistakes.append({
                "turn": t,
                "severity": "CRITICAL",
                "category": "Empty Bench",
                "detail": f"P0 has NO bench Pokemon. One KO = instant loss.",
                "reward_fix": "Apply bench=0 penalty: r_bench -= 2.0. Reward benching early.",
            })

        # Mistake: No energy attached for multiple early turns
        if t in range(1, 6) and t not in p0_energy_turns:
            if active and active["energy_count"] == 0:
                mistakes.append({
                    "turn": t,
                    "severity": "HIGH",
                    "category": "No Energy Attachment",
                    "detail": f"Turn {t}: Active [{active['name']}] has 0 energy. Agent skipped energy attachment.",
                    "reward_fix": "r_no_energy penalty (-1.0/step) forces energy attachment. Raise r_en to 1.5.",
                })

        # Mistake: Never attacked despite having energy
        if t >= 3 and t not in p0_attacked_turns and active and active["energy_count"] >= 2:
            mistakes.append({
                "turn": t,
                "severity": "HIGH",
                "category": "Passive Play (No Attack)",
                "detail": f"Turn {t}: Active [{active['name']}] had {active['energy_count']} energy but did not attack.",
                "reward_fix": "Raise attack reward. Boost prize_taken weight to make attacking attractive.",
            })

        # Mistake: Opponent bench is full but P0 has no bench backup
        if p1["bench_count"] >= 3 and p0["bench_count"] == 0:
            mistakes.append({
                "turn": t,
                "severity": "HIGH",
                "category": "Board Asymmetry",
                "detail": f"Turn {t}: P1 bench={p1['bench_count']} vs P0 bench=0. Opponent completely outnumbers P0.",
                "reward_fix": "Bench Pokemon immediately. Miraidon ability or search trainers early.",
            })

    # Deduplicate by (category, turn)
    seen_keys = set()
    deduped = []
    for m in mistakes:
        k = (m["category"], m["turn"])
        if k not in seen_keys:
            seen_keys.add(k)
            deduped.append(m)

    return deduped


# ── Report Writer ──────────────────────────────────────────────────────────

def write_report(data, names, events, prizes, snapshots, mistakes, output_path):
    episode_id = data.get("info", {}).get("EpisodeId", "unknown")
    agents = data.get("info", {}).get("Agents", [{}, {}])
    p0_name = agents[0].get("Name", "P0") if agents else "P0"
    p1_name = agents[1].get("Name", "P1") if len(agents) > 1 else "P1"
    rewards = data.get("rewards", [0, 0])
    p0_result = "WIN" if rewards[0] > 0 else "LOSS" if rewards[0] < 0 else "DRAW"
    p1_result = "WIN" if rewards[1] > 0 else "LOSS" if rewards[1] < 0 else "DRAW"

    result_event = next((e for e in events if e["type"] == "Result"), None)
    end_reason = RESULT_REASONS.get(result_event["reason"], "Unknown") if result_event else "Unknown"

    final_snap = snapshots[-1] if snapshots else {}
    total_turns = final_snap.get("turn", 0)

    p0_attacks = [e for e in events if e["type"] == "Attack" and e["player"] == 0]
    p1_attacks = [e for e in events if e["type"] == "Attack" and e["player"] == 1]
    p0_energy = [e for e in events if e["type"] == "Attach" and e["player"] == 0]
    p1_energy = [e for e in events if e["type"] == "Attach" and e["player"] == 1]
    p1_evolves = [e for e in events if e["type"] == "Evolve" and e["player"] == 1]

    p0_prizes_taken = sum(p["p0_took"] for p in prizes)
    p1_prizes_taken = sum(p["p1_took"] for p in prizes)

    lines = []
    lines.append(f"# Battle Post-Mortem Report -- Episode {episode_id}\n")
    lines.append(f"> Generated by `Learn_from_mistake/analyze_loss.py`\n")
    lines.append("")

    lines.append("## Match Summary\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Episode | {episode_id} |")
    lines.append(f"| **{p0_name} (P0)** | **{p0_result}** |")
    lines.append(f"| **{p1_name} (P1)** | **{p1_result}** |")
    lines.append(f"| Total Turns | {total_turns} |")
    lines.append(f"| End Reason | {end_reason} |")
    lines.append(f"| P0 Prizes Taken | {p0_prizes_taken} / 6 |")
    lines.append(f"| P1 Prizes Taken | {p1_prizes_taken} / 6 |")
    lines.append("")

    lines.append("## Prize Progression\n")
    lines.append("| Turn | P0 Prizes Left | P1 Prizes Left | Event |")
    lines.append("|---|---|---|---|")
    for p in prizes:
        note = ""
        if p["p0_took"] > 0:
            note = f"P0 took {p['p0_took']} prize(s)"
        if p["p1_took"] > 0:
            note = f"P1 took {p['p1_took']} prize(s)"
        lines.append(f"| {p['turn']} | {p['p0_prizes']} | {p['p1_prizes']} | {note} |")
    lines.append("")

    lines.append("## Key Events Timeline\n")
    lines.append("| Turn | Player | Event | Detail |")
    lines.append("|---|---|---|---|")
    for e in events:
        t = e["turn"]
        p = e.get("player", "?")
        label = f"P{p}" if isinstance(p, int) else str(p)
        if e["type"] == "Attack":
            lines.append(f"| {t} | {label} | ATTACK | [{e['card']}] |")
        elif e["type"] == "HpChange":
            lines.append(f"| {t} | {label} | HP CHANGE | [{e['card']}] {e['value']:+d} |")
        elif e["type"] == "Evolve":
            lines.append(f"| {t} | {label} | EVOLVE | {e['from']} -> {e['to']} |")
        elif e["type"] == "Attach":
            lines.append(f"| {t} | {label} | ENERGY | [{e['energy']}] on [{e['target']}] |")
        elif e["type"] == "Switch":
            lines.append(f"| {t} | {label} | SWITCH | {e['from']} -> {e['to']} |")
        elif e["type"] == "Result":
            reason = RESULT_REASONS.get(e.get("reason"), "Unknown")
            lines.append(f"| {t} | -- | GAME OVER | {reason} |")
    lines.append("")

    lines.append("## Board State Per Turn\n")
    lines.append("| Turn | P0 Active | P0 HP | P0 Energy | P0 Bench | P1 Active | P1 Bench |")
    lines.append("|---|---|---|---|---|---|---|")
    for snap in snapshots:
        t = snap["turn"]
        p0 = snap["p0"]
        p1 = snap["p1"]
        a0 = p0["active"]
        a1 = p1["active"]
        a0_name = a0["name"] if a0 else "none"
        a0_hp = f"{a0['hp']}/{a0['max_hp']}" if a0 else "-"
        a0_en = a0["energy_count"] if a0 else 0
        a1_name = a1["name"] if a1 else "none"
        p0_bench = ", ".join(p0["bench_names"]) or "--"
        p1_bench = ", ".join(p1["bench_names"]) or "--"
        lines.append(f"| {t} | {a0_name} | {a0_hp} | {a0_en} | {p0_bench} | {a1_name} | {p1_bench} |")
    lines.append("")

    lines.append("## Mistake Analysis\n")
    if not mistakes:
        lines.append("No critical mistakes detected.\n")
    else:
        critical = [m for m in mistakes if m["severity"] == "CRITICAL"]
        high = [m for m in mistakes if m["severity"] == "HIGH"]

        if critical:
            lines.append("### Critical Mistakes\n")
            for m in critical:
                lines.append(f"- **Turn {m['turn']} | {m['category']}**: {m['detail']}")
            lines.append("")

        if high:
            lines.append("### High-Severity Mistakes\n")
            for m in high:
                lines.append(f"- **Turn {m['turn']} | {m['category']}**: {m['detail']}")
            lines.append("")

    lines.append("## RL Reward Shaping Recommendations\n")
    seen_fixes = set()
    fix_lines = []
    for m in mistakes:
        fix = m["reward_fix"]
        if fix not in seen_fixes:
            seen_fixes.add(fix)
            fix_lines.append(f"- {fix}")

    standard = [
        "**Energy first**: `r_en = 1.5` per energy attached. `r_no_energy = -1.0/step` when no energy attached yet.",
        "**Bench safety**: `r_bench = -1.0` when bench=0 (single point of failure). `r_bench += 5.0` on bench recovery.",
        "**Stall penalty**: `r_stall = -0.15/step` to discourage passive turns.",
        "**Prize urgency**: `r_prize_taken = 12.0` to make attacking the highest non-terminal reward.",
        "**Opponent modeling**: Use dynamic opponent deck identification to improve MCTS rollout accuracy.",
    ]
    for s in standard:
        if s not in seen_fixes:
            fix_lines.append(f"- {s}")

    lines.extend(fix_lines)
    lines.append("")

    lines.append("## Action Statistics\n")
    lines.append("| Metric | P0 (Suchai) | P1 (Opponent) |")
    lines.append("|---|---|---|")
    lines.append(f"| Total Attacks | {len(p0_attacks)} | {len(p1_attacks)} |")
    lines.append(f"| Energy Attachments | {len(p0_energy)} | {len(p1_energy)} |")
    lines.append(f"| Prizes Taken | {p0_prizes_taken} | {p1_prizes_taken} |")
    lines.append(f"| Evolutions | 0 | {len(p1_evolves)} |")
    lines.append("")

    lines.append("## Root Cause Summary\n")
    if p0_prizes_taken == 0:
        lines.append("> [!CAUTION]")
        lines.append("> **P0 took ZERO prizes.** The agent never attacked effectively.")
        lines.append("> This is a complete setup failure -- energy and bench management must be fixed first.")
    elif p0_prizes_taken < 3:
        lines.append("> [!WARNING]")
        lines.append(f"> **P0 only took {p0_prizes_taken} prize(s).** Aggression was too low.")
    else:
        lines.append("> [!NOTE]")
        lines.append(f"> **P0 took {p0_prizes_taken} prize(s)** -- competitive but ultimately outplayed.")
    lines.append("")

    output = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Report saved to: {output_path}")
    return output_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_loss.py <replay.json> [output.md]")
        sys.exit(1)

    replay_path = sys.argv[1]
    if not os.path.exists(replay_path):
        print(f"Error: replay file not found: {replay_path}")
        sys.exit(1)

    data = load_replay(replay_path)
    names = build_card_names(data)
    events = extract_events(data, names)
    prizes = extract_prize_progression(data)
    snapshots = extract_turn_snapshots(data, names)

    rewards = data.get("rewards", [0, 0])
    losing_player = 0 if rewards[0] < 0 else 1
    mistakes = detect_mistakes(events, snapshots, losing_player)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        episode_id = data.get("info", {}).get("EpisodeId", Path(replay_path).stem)
        script_dir = Path(__file__).parent
        output_path = str(script_dir / f"{episode_id}_report.md")

    write_report(data, names, events, prizes, snapshots, mistakes, output_path)


if __name__ == "__main__":
    main()
