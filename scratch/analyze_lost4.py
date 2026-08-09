import json
import glob
import os

lost4_dir = "lost4"
json_files = glob.glob(os.path.join(lost4_dir, "*.json"))

print(f"Parsing {len(json_files)} replay JSONs in {lost4_dir}/\n")

retreat_events = []

for filepath in json_files:
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data if isinstance(data, list) else data.get("steps", [])
    info = data.get("info", {}) if isinstance(data, dict) else {}
    team_names = info.get("TeamNames", ["Player 0", "Player 1"])
    
    agent_idx = 0
    # Determine agent player index (Mewtwo ex deck cards)
    for s in steps[:5]:
        if not isinstance(s, list) or len(s) <= 1:
            continue
        for p_i in (0, 1):
            obs = s[p_i].get("observation", {})
            if isinstance(obs, dict) and isinstance(obs.get("current"), dict):
                players = obs["current"].get("players", [])
                if isinstance(players, (list, tuple)) and len(players) > p_i:
                    h = [c.get("id", -1) for c in players[p_i].get("hand", []) if isinstance(c, dict)]
                    if any(cid in (431, 400, 401, 414, 434) for cid in h):
                        agent_idx = p_i
                        break

    for i, step in enumerate(steps):
        if not isinstance(step, list) or len(step) <= agent_idx:
            continue
        s_data = step[agent_idx]
        action = s_data.get("action")
        obs = s_data.get("observation", {})
        if not isinstance(obs, dict) or not isinstance(obs.get("current"), dict):
            continue
            
        curr = obs["current"]
        turn = curr.get("turn", 0)
        select = obs.get("select", {})
        options = select.get("option", [])
        
        # Check if action selected RETREAT (12)
        if isinstance(action, list) and len(action) > 0 and action[0] is not None and isinstance(options, list):
            sel_idx = action[0] if isinstance(action[0], int) else (action[0][0] if isinstance(action[0], list) and len(action[0]) > 0 else -1)
            if isinstance(sel_idx, int) and 0 <= sel_idx < len(options):
                chosen_opt = options[sel_idx]
                if isinstance(chosen_opt, dict) and chosen_opt.get("type") == 12:
                    p_me = curr["players"][agent_idx]
                    p_opp = curr["players"][1 - agent_idx]
                    
                    act_me = p_me.get("active", [])
                    act_pk = act_me[0] if (act_me and isinstance(act_me[0], dict)) else {}
                    act_name = act_pk.get("name", str(act_pk.get("id", -1)))
                    act_id = act_pk.get("id", -1)
                    act_hp = act_pk.get("hp", 0)
                    act_max_hp = act_pk.get("maxHp", 100)
                    energies = act_pk.get("energyCards", [])
                    act_en_cnt = len(energies) if isinstance(energies, list) else 0
                    
                    has_attack_opt = any(o.get("type") == 13 for o in options if isinstance(o, dict))
                    
                    retreat_events.append({
                        "file": fname,
                        "step": i,
                        "turn": turn,
                        "active": act_name,
                        "act_id": act_id,
                        "hp": f"{act_hp}/{act_max_hp}",
                        "energy": act_en_cnt,
                        "has_attack_opt": has_attack_opt,
                        "action": action
                    })

print("="*105)
print(f"{'FILE':<18} | {'STEP':<5} | {'TURN':<5} | {'ACTIVE PK':<22} | {'HP':<10} | {'ENERGY':<8} | {'ATTACK OPTION LEGAL?'}")
print("="*105)

for r in retreat_events:
    print(f"{r['file']:<18} | {r['step']:<5} | {r['turn']:<5} | {r['active']:<22} | {r['hp']:<10} | {r['energy']:<8} | {r['has_attack_opt']}")

print("="*105)
print(f"Total RETREAT actions in lost4: {len(retreat_events)}")
