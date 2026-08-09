import json
import glob
import os

iono_dir = "iono"
json_files = glob.glob(os.path.join(iono_dir, "*.json"))

for filepath in json_files:
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data if isinstance(data, list) else data.get("steps", [])
    
    # Identify agent player index (Mewtwo deck)
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

    prev_active = None
    prev_energy = 0
    
    print("="*90)
    print(f"TRACING MATCH: {fname} (Agent Index: {agent_idx})")
    print("="*90)
    
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
        p_me = curr["players"][agent_idx]
        
        act_list = p_me.get("active", [])
        act_pk = act_list[0] if (act_list and isinstance(act_list[0], dict)) else None
        act_name = act_pk.get("name", str(act_pk.get("id", -1))) if act_pk else "None"
        energies = act_pk.get("energyCards", []) if act_pk else []
        energy_cnt = len(energies) if isinstance(energies, list) else 0
        
        opts = obs.get("select", {}).get("option", [])
        
        # Check if active changed
        if prev_active is not None and act_name != prev_active:
            # Active changed!
            opt_strs = []
            if isinstance(opts, list):
                for o in opts[:5]:
                    if isinstance(o, dict):
                        opt_strs.append(f"{o.get('type')}:{o.get('name', o.get('id', ''))}")
            
            print(f"Step {i:03d} | Turn {turn:02d} | ACTIVE CHANGED: {prev_active} ({prev_energy} energy) -> {act_name} ({energy_cnt} energy) | Action: {action} | Opts: {opt_strs}")

        prev_active = act_name
        prev_energy = energy_cnt
