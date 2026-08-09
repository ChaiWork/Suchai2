import json
import glob
import os

iono_dir = "iono"
json_files = glob.glob(os.path.join(iono_dir, "*.json"))

print(f"Inspecting {len(json_files)} replay JSONs in {iono_dir}/ for RETREAT actions...\n")

retreat_events = []

for filepath in json_files:
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if isinstance(data, list):
        steps = data
        info = {}
        team_names = ["Player 0", "Player 1"]
    else:
        info = data.get("info", {})
        team_names = info.get("TeamNames", ["Player 0", "Player 1"])
        steps = data.get("steps", [])
    
    # Identify agent player index
    agent_idx = 0
    for s in steps[:5]:
        if isinstance(s, list) and len(s) > 1:
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
        
        # Check if action corresponds to RETREAT option (type 12)
        if isinstance(action, list) and len(action) > 0 and isinstance(options, list):
            sel_idx = action[0]
            if 0 <= sel_idx < len(options):
                chosen_opt = options[sel_idx]
                if isinstance(chosen_opt, dict) and chosen_opt.get("type") == 12:
                    # RETREAT chosen!
                    p_me = curr["players"][agent_idx]
                    p_opp = curr["players"][1 - agent_idx]
                    
                    act_me = p_me.get("active", [])
                    act_pk = act_me[0] if (act_me and isinstance(act_me[0], dict)) else {}
                    act_name = act_pk.get("name", str(act_pk.get("id", -1)))
                    act_hp = act_pk.get("hp", 0)
                    act_max_hp = act_pk.get("maxHp", 100)
                    energies = act_pk.get("energyCards", [])
                    act_energy_cnt = len(energies) if isinstance(energies, list) else 0
                    
                    # Check if ATTACK was also legally available in options!
                    has_attack_opt = any(o.get("type") == 13 for o in options if isinstance(o, dict))
                    
                    retreat_events.append({
                        "file": fname,
                        "step": i,
                        "turn": turn,
                        "active": act_name,
                        "hp": f"{act_hp}/{act_max_hp}",
                        "energy": act_energy_cnt,
                        "has_attack_option": has_attack_opt,
                        "num_options": len(options)
                    })

print("="*95)
print(f"{'FILE':<20} | {'STEP':<5} | {'TURN':<5} | {'ACTIVE PK':<20} | {'HP':<10} | {'ENERGY':<8} | {'ATTACK OPTION LEGAL?'}")
print("="*95)

for r in retreat_events:
    print(f"{r['file']:<20} | {r['step']:<5} | {r['turn']:<5} | {r['active']:<20} | {r['hp']:<10} | {r['energy']:<8} | {r['has_attack_option']}")

print("="*95)
print(f"\nTotal RETREAT actions observed across 10 matches: {len(retreat_events)}")
has_attack_retreats = sum(1 for r in retreat_events if r['has_attack_option'])
print(f"Retreats performed while ATTACK option was legally available: {has_attack_retreats} / {len(retreat_events)} ({has_attack_retreats/max(1, len(retreat_events))*100:.1f}%)")
