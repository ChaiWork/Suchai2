import json
import os

json_path = r"d:\codingProject\pokemon-tcg-ai-agent\pokemon-tcg-ai-battle-challenge-strategy\86125248.json"

if not os.path.exists(json_path):
    print(f"Error: {json_path} does not exist.")
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Episode ID: {data.get('info', {}).get('EpisodeId')}")
print(f"Rewards: {data.get('rewards')}")
print(f"Statuses: {data.get('statuses')}")

steps = data.get("steps", [])
print(f"Total steps in replay: {len(steps)}")

# Parse logs and actions step by step
all_logs = []
for step_idx, step_data in enumerate(steps):
    # Each step contains a list of states/actions for each agent (usually 2 agents)
    for agent_idx, agent_step in enumerate(step_data):
        logs = agent_step.get("logs", [])
        for log in logs:
            all_logs.append((step_idx, agent_idx, log))
            
        action = agent_step.get("action")
        select = agent_step.get("select")
        if action and select and select.get("option"):
            options = select.get("option")
            chosen_opts = [options[a] for a in action if a < len(options)]
            # Print action selections of interest
            print(f"Step {step_idx:03d} | Agent {agent_idx} | Context: {select.get('context')} | Action: {action} | Chosen: {chosen_opts}")

print("\n--- Game Logs Summary ---")
card_names = {}
# Scan for all card metadata to map IDs to names
for step_data in steps:
    for agent_step in step_data:
        current = agent_step.get("current")
        if current and current.get("players"):
            for p in current["players"]:
                for card_list in [p.get("deck", []), p.get("hand", []), p.get("discard", []), p.get("active", []), p.get("bench", [])]:
                    if card_list:
                        for card in card_list:
                            if card and "id" in card and "name" in card:
                                card_names[card["id"]] = card["name"]

print(f"Loaded card name mappings for {len(card_names)} cards.")

for step_idx, agent_idx, log in all_logs:
    log_type = log.get("type")
    
    # Map card IDs to names in log for readability
    if "cardId" in log and log["cardId"] in card_names:
        log["card_name"] = card_names[log["cardId"]]
        
    # Standard log representations
    if log_type == "MoveCard":
        from_area = log.get("fromArea")
        to_area = log.get("toArea")
        card_name = log.get("card_name", f"Card ID {log.get('cardId')}")
        print(f"  [Step {step_idx:03d}] Player {log.get('playerIndex')}: Moved '{card_name}' from {from_area} to {to_area}")
    elif log_type == "Attack":
        print(f"  [Step {step_idx:03d}] Player {log.get('playerIndex')}: Attacked with attack ID {log.get('attackId')} (damage: {log.get('damage')})")
    elif log_type == "Knockout":
        print(f"  [Step {step_idx:03d}] Player {log.get('playerIndex')}: Pokémon knocked out!")
    elif log_type == "Prize":
        print(f"  [Step {step_idx:03d}] Player {log.get('playerIndex')}: Took prize card!")
    elif log_type == "Win":
        print(f"  [Step {step_idx:03d}] Player {log.get('playerIndex')}: WON THE GAME!")
    else:
        print(f"  [Step {step_idx:03d}] Log Type '{log_type}': {log}")
