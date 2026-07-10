from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ZONE_NAME = {
    1: "Deck/Prize",
    2: "Hand",
    3: "Discard",
    4: "Active",
    5: "Bench",
    6: "Prize",
}

LOG_TYPE_NAME = {
    0: "Shuffle",
    1: "HasBasicPokemon",
    2: "SetActive",
    3: "SetBench",
    4: "Damage",
    5: "Draw",
    6: "MoveCard",
    7: "Heal",
    8: "Attach",
    9: "Detach",
    10: "KnockOut",
    11: "Play",
    12: "Evolve",
    13: "UseAbility",
    14: "Retreat",
    15: "Attack",
    16: "TakePrize",
    "Shuffle": "Shuffle",
    "HasBasicPokemon": "HasBasicPokemon",
    "Draw": "Draw",
    "MoveCard": "MoveCard",
}

OPTION_TYPE_NAME = {
    0: "PLAY",
    1: "ATTACH",
    2: "EVOLVE",
    3: "CARD",
    4: "ABILITY",
    5: "RETREAT",
    6: "ATTACK",
    7: "ATTACK",
    8: "PLAY",
    9: "YES",
    10: "NO",
    11: "ABILITY",
    12: "RETREAT",
    13: "ATTACH",
    14: "END",
    "Yes": "YES",
    "No": "NO",
    "Card": "CARD",
}


def _collect_card_ids(obj: Any, out: Optional[set] = None) -> set:
    if out is None:
        out = set()
    if isinstance(obj, dict):
        if "id" in obj:
            try:
                out.add(int(obj["id"]))
            except Exception:
                pass
        for v in obj.values():
            _collect_card_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_card_ids(v, out)
    return out


def extract_card_image_page_map(pdf_path: str | Path) -> Dict[int, int]:
    """
    Card_ID_List_*.pdf の一覧ページにある内部リンクを読んで、
    Card ID -> カード画像ページ番号 の辞書を作ります。
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF is required. Run: pip install pymupdf") from e

    doc = fitz.open(str(pdf_path))
    mapping: Dict[int, int] = {}
    ordered_targets: List[int] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        links = [l for l in page.get_links() if "page" in l and l.get("page") is not None]
        if not links:
            continue

        words = page.get_text("words")  # x0, y0, x1, y1, text, ...
        links = sorted(links, key=lambda l: (float(l["from"].y0), float(l["from"].x0)))

        for link in links:
            rect = link["from"]
            y_mid = (float(rect.y0) + float(rect.y1)) / 2
            target = int(link["page"])
            ordered_targets.append(target)

            # 同じ行の左端にある数値を Card ID とみなす
            row_words = []
            for w in words:
                x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
                wy = (y0 + y1) / 2
                if x0 < 145 and abs(wy - y_mid) < 8 and text.strip().isdigit():
                    row_words.append((abs(wy - y_mid), x0, text.strip()))

            if row_words:
                row_words.sort()
                try:
                    card_id = int(row_words[0][2])
                    mapping[card_id] = target
                except Exception:
                    pass

    # 万一テキスト抽出が外れても、リンク順がCard ID順なら救済
    if len(mapping) < max(10, len(ordered_targets) // 2):
        fallback = {i + 1: p for i, p in enumerate(ordered_targets)}
        mapping.update({k: v for k, v in fallback.items() if k not in mapping})

    return mapping


def _crop_card_image(img):
    """
    PDFのカード画像ページは白い余白が大きいので、非白領域をざっくり切り出す。
    """
    from PIL import Image, ImageChops

    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if not bbox:
        return img

    left, top, right, bottom = bbox

    pad_x = max(8, int((right - left) * 0.04))
    pad_y = max(8, int((bottom - top) * 0.04))

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(img.width, right + pad_x)
    bottom = min(img.height, bottom + pad_y)

    return img.crop((left, top, right, bottom))


def prepare_card_images(
    pdf_path: str | Path,
    out_dir: str | Path = "card_images",
    card_ids: Optional[List[int]] = None,
    zoom: float = 2.0,
    overwrite: bool = False,
) -> Dict[int, str]:
    """
    PDF後半のカード画像ページをPNGとして切り出します。
    """
    try:
        import fitz
        from PIL import Image
    except ImportError as e:
        raise ImportError("PyMuPDF and Pillow are required. Run: pip install pymupdf pillow") from e

    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_map = extract_card_image_page_map(pdf_path)
    if card_ids is None:
        target_ids = sorted(page_map)
    else:
        target_ids = sorted({int(x) for x in card_ids if int(x) in page_map})

    doc = fitz.open(str(pdf_path))
    saved: Dict[int, str] = {}

    for card_id in target_ids:
        out_path = out_dir / f"{card_id}.png"
        if out_path.exists() and not overwrite:
            saved[card_id] = str(out_path).replace(os.sep, "/")
            continue

        page_no = page_map.get(card_id)
        if page_no is None:
            continue

        page = doc[page_no]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = _crop_card_image(img)
        img.save(out_path)

        saved[card_id] = str(out_path).replace(os.sep, "/")

    return saved


def _existing_card_images(card_image_dir: str | Path, html_dir: Optional[Path] = None) -> Dict[int, str]:
    d = Path(card_image_dir)
    if not d.exists():
        return {}
    out: Dict[int, str] = {}
    for p in d.glob("*.png"):
        try:
            cid = int(p.stem)
        except Exception:
            continue
        if html_dir is not None:
            try:
                rel = os.path.relpath(p, html_dir)
            except Exception:
                rel = str(p)
            out[cid] = rel.replace(os.sep, "/")
        else:
            out[cid] = str(p).replace(os.sep, "/")
    return out


def _safe_name(card: Optional[Dict[str, Any]]) -> str:
    if not card:
        return ""
    return str(card.get("name") or f"Card {card.get('id', '?')}")


def _card(card: Optional[Dict[str, Any]], image_map: Optional[Dict[int, str]] = None) -> Optional[Dict[str, Any]]:
    if not card:
        return None
    energies = card.get("energies") or []
    energy_cards = card.get("energyCards") or []
    tools = card.get("tools") or []
    cid = card.get("id")
    try:
        cid_int = int(cid)
    except Exception:
        cid_int = None
    image = image_map.get(cid_int) if image_map and cid_int is not None else None
    return {
        "id": card.get("id"),
        "serial": card.get("serial"),
        "name": _safe_name(card),
        "hp": card.get("hp"),
        "maxHp": card.get("maxHp"),
        "energies": len(energies) if isinstance(energies, list) else 0,
        "energyCards": [_safe_name(c) for c in energy_cards] if isinstance(energy_cards, list) else [],
        "tools": [_safe_name(c) for c in tools] if isinstance(tools, list) else [],
        "appearThisTurn": bool(card.get("appearThisTurn")),
        "preEvolution": [_safe_name(c) for c in (card.get("preEvolution") or [])],
        "image": image,
    }


def _cards(cards: Any, image_map: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
    if not isinstance(cards, list):
        return []
    return [c for c in (_card(x, image_map) for x in cards) if c]


def _player(p: Dict[str, Any], i: int, team_names: List[str], image_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    hand = p.get("hand")
    visible_hand = _cards(hand, image_map) if isinstance(hand, list) else None
    return {
        "index": i,
        "name": team_names[i] if i < len(team_names) else f"Player {i}",
        "active": _cards(p.get("active"), image_map),
        "bench": _cards(p.get("bench"), image_map),
        "hand": visible_hand,
        "handCount": p.get("handCount"),
        "deckCount": p.get("deckCount"),
        "discardCount": len(p.get("discard") or []),
        "prizeCount": len(p.get("prize") or []),
        "status": {
            "asleep": bool(p.get("asleep")),
            "burned": bool(p.get("burned")),
            "confused": bool(p.get("confused")),
            "paralyzed": bool(p.get("paralyzed")),
            "poisoned": bool(p.get("poisoned")),
        },
    }


def _log_text(log: Dict[str, Any], card_names: Dict[int, str]) -> str:
    t = log.get("type")
    label = LOG_TYPE_NAME.get(t, str(t))
    pid = log.get("playerIndex")
    who = f"P{pid}" if pid is not None else ""
    cid = log.get("cardId")
    name = card_names.get(cid, f"Card {cid}") if cid is not None else ""

    if label == "Draw":
        return f"{who} draws {name}"
    if label == "MoveCard":
        fa = ZONE_NAME.get(log.get("fromArea"), str(log.get("fromArea")))
        ta = ZONE_NAME.get(log.get("toArea"), str(log.get("toArea")))
        return f"{who} moves {name}: {fa} → {ta}"
    if label == "Damage":
        dmg = log.get("damage") or log.get("amount") or log.get("value") or ""
        return f"{who} damage {name} {dmg}".strip()
    if label == "Heal":
        return f"{who} heals {name}"
    if label == "KnockOut":
        return f"{who} knock out {name}"
    if label == "Attack":
        return f"{who} attacks with {name}"
    if label == "Evolve":
        return f"{who} evolves {name}"
    if label == "Attach":
        return f"{who} attaches {name}"
    if label == "Play":
        return f"{who} plays {name}"
    if label == "HasBasicPokemon":
        return f"{who} has basic Pokémon: {log.get('hasBasicPokemon')}"
    if label == "Shuffle":
        return f"{who} shuffles"
    return f"{who} {label} {name}".strip()


def _collect_card_names(obj: Any, out: Dict[int, str]) -> None:
    if isinstance(obj, dict):
        if "id" in obj and "name" in obj:
            out[obj["id"]] = obj["name"]
        for v in obj.values():
            _collect_card_names(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_card_names(v, out)


def _option_text(opt: Dict[str, Any], current: Optional[Dict[str, Any]]) -> str:
    t = opt.get("type")
    label = OPTION_TYPE_NAME.get(t, str(t))
    area = opt.get("area")
    index = opt.get("index")
    player = opt.get("playerIndex")
    card_name = ""

    try:
        if current and area is not None and index is not None and player is not None:
            p = current["players"][player]
            if area == 2 and isinstance(p.get("hand"), list) and index < len(p["hand"]):
                card_name = _safe_name(p["hand"][index])
            elif area == 4 and isinstance(p.get("active"), list) and index < len(p["active"]):
                card_name = _safe_name(p["active"][index])
            elif area == 5 and isinstance(p.get("bench"), list) and index < len(p["bench"]):
                card_name = _safe_name(p["bench"][index])
    except Exception:
        pass

    suffix = f" · {card_name}" if card_name else ""
    return f"{label}{suffix}"


def _pick_observation(step: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = []
    for agent_index, entry in enumerate(step):
        obs = entry.get("observation") or {}
        if obs.get("current") is not None:
            candidates.append((entry.get("status") == "ACTIVE", agent_index, obs, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (not x[0], x[1]))
    active, agent_index, obs, entry = candidates[0]
    return {"agentIndex": agent_index, "obs": obs, "entry": entry}


def build_battle_data(json_path: str | Path, image_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    path = Path(json_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    team_names = raw.get("info", {}).get("TeamNames") or [
        a.get("Name", f"Player {i}") for i, a in enumerate(raw.get("info", {}).get("Agents", []))
    ]
    card_names: Dict[int, str] = {}
    _collect_card_names(raw, card_names)

    frames = []
    for step_index, step in enumerate(raw.get("steps", [])):
        picked = _pick_observation(step)
        if not picked:
            continue

        obs = picked["obs"]
        current = obs.get("current")
        if not current:
            continue

        players = current.get("players") or []
        frame = {
            "step": obs.get("step", step_index),
            "rawStep": step_index,
            "turn": current.get("turn"),
            "turnActionCount": current.get("turnActionCount"),
            "viewer": current.get("yourIndex"),
            "firstPlayer": current.get("firstPlayer"),
            "result": current.get("result"),
            "stadium": _cards(current.get("stadium"), image_map),
            "energyAttached": bool(current.get("energyAttached")),
            "supporterPlayed": bool(current.get("supporterPlayed")),
            "stadiumPlayed": bool(current.get("stadiumPlayed")),
            "players": [_player(p, i, team_names, image_map) for i, p in enumerate(players)],
            "logs": [_log_text(l, card_names) for l in (obs.get("logs") or [])][-30:],
            "logCount": len(obs.get("logs") or []),
            "select": None,
            "action": picked["entry"].get("action"),
            "status": picked["entry"].get("status"),
        }

        sel = obs.get("select")
        if isinstance(sel, dict):
            options = sel.get("option") or []
            frame["select"] = {
                "type": sel.get("type"),
                "context": sel.get("context"),
                "minCount": sel.get("minCount"),
                "maxCount": sel.get("maxCount"),
                "options": [_option_text(o, current) for o in options],
            }

        frames.append(frame)

    return {
        "title": raw.get("title") or raw.get("name") or "Pokémon TCG Battle",
        "episodeId": raw.get("info", {}).get("EpisodeId"),
        "teamNames": team_names,
        "rewards": raw.get("rewards"),
        "statuses": raw.get("statuses"),
        "frames": frames,
    }


HTML_TEMPLATE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pokémon TCG Battle Viewer</title><style>
:root{--bg:#0f131b;--panel:#181d29;--panel2:#202737;--text:#eef3ff;--muted:#aab3c5;--accent:#ffd166;--danger:#ef476f;--ok:#06d6a0;--line:rgba(255,255,255,.10);--p0:#5fb3ff;--p1:#ff8a8a}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#26304a 0,var(--bg) 48%);color:var(--text);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.app{max-width:980px;margin:0 auto;padding:7px}.header{display:flex;justify-content:space-between;gap:8px;align-items:start;margin-bottom:5px}.title{font-size:16px;font-weight:850;line-height:1.1}.sub{color:var(--muted);font-size:10px;margin-top:2px}.badge{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:4px 7px;color:var(--muted);background:rgba(0,0,0,.18);font-size:10px;white-space:nowrap}.arena{display:grid;grid-template-rows:auto auto auto;gap:5px}.player{background:rgba(24,29,41,.92);border:1px solid var(--line);border-radius:13px;padding:6px;box-shadow:0 8px 22px rgba(0,0,0,.20)}.player.p0{border-left:4px solid var(--p0)}.player.p1{border-left:4px solid var(--p1)}.player.opponent{transform:rotate(180deg)}.player.opponent .inner{transform:rotate(180deg)}.player-head{display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:4px}.player-name{font-size:13px;font-weight:850;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.meta-line{color:var(--muted);font-size:9px;white-space:nowrap}.counts{display:flex;gap:3px;flex-wrap:wrap}.counts span{background:rgba(0,0,0,.18);border:1px solid var(--line);border-radius:7px;padding:2px 5px;text-align:center;font-size:9px;color:var(--muted);min-width:36px}.counts b{display:inline;color:var(--text);font-size:11px;margin-left:2px}.field{display:grid;grid-template-columns:minmax(108px,1fr) minmax(0,2fr);gap:5px;align-items:stretch}.zone-title{color:var(--accent);font-size:8px;font-weight:850;text-transform:uppercase;letter-spacing:.06em;margin:0 0 2px}.active-zone{min-height:130px;display:flex;align-items:stretch}.bench{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:3px}.card{background:linear-gradient(180deg,rgba(255,255,255,.10),rgba(255,255,255,.03));border:1px solid var(--line);border-radius:9px;padding:3px;min-height:92px;position:relative;overflow:hidden;cursor:pointer}.card.active{width:100%;min-height:130px;border-color:rgba(255,209,102,.65)}.card.empty{opacity:.20;border-style:dashed;cursor:default}.thumb{width:100%;height:64px;object-fit:contain;display:block;border-radius:6px;background:rgba(255,255,255,.04);margin-bottom:2px}.card.active .thumb{height:92px}.card-id{color:var(--muted);font-size:7px}.card-name{font-size:8px;font-weight:850;line-height:1.05;margin:1px 0 2px;word-break:break-word}.card.active .card-name{font-size:10px}.hp-row{display:flex;justify-content:space-between;font-size:7px;color:var(--muted)}.hpbar{height:4px;background:rgba(255,255,255,.10);border-radius:999px;overflow:hidden;margin-top:1px}.hpfill{height:100%;background:linear-gradient(90deg,var(--ok),var(--accent),var(--danger));border-radius:999px}.energy{margin-top:2px;font-size:7px;color:var(--accent);line-height:1.05}.energy-list{color:#ffe7a3}.tools{margin-top:2px;font-size:7px;color:var(--muted);line-height:1.1}.hand-strip{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:3px;margin-top:5px}.hand-card{min-height:72px}.hand-hidden{color:var(--muted);font-size:10px;border:1px dashed var(--line);border-radius:999px;padding:5px;text-align:center}.center{background:rgba(20,24,35,.78);border:1px solid var(--line);border-radius:13px;padding:4px 7px;display:flex;align-items:center;justify-content:space-between;gap:8px}.big-turn{font-size:18px;font-weight:950;white-space:nowrap}.step-line{color:var(--muted);font-size:9px}.flags{display:flex;gap:3px;flex-wrap:wrap;justify-content:flex-end}.flag{background:rgba(255,255,255,.06);border:1px solid var(--line);border-radius:999px;padding:2px 6px;font-size:9px;color:var(--muted)}.bottom{margin-top:5px;background:rgba(24,29,41,.92);border:1px solid var(--line);border-radius:13px;padding:6px}.controls{display:grid;grid-template-columns:auto auto auto 1fr auto;gap:4px;align-items:center;margin-bottom:5px}button{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:5px 8px;border-radius:999px;cursor:pointer;font-weight:800;font-size:11px}button:hover{filter:brightness(1.16)}input[type=range]{width:100%;min-width:70px}.tabs{display:grid;grid-template-columns:1fr 1fr;gap:5px}.panel{background:rgba(255,255,255,.035);border:1px solid var(--line);border-radius:11px;padding:5px;min-height:72px}.panel-title{color:var(--accent);font-size:9px;font-weight:850;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}.options{display:flex;flex-direction:column;gap:3px;max-height:126px;overflow:auto}.option{border:1px solid var(--line);border-radius:8px;padding:4px 5px;background:rgba(255,255,255,.04);font-size:10px}.log{max-height:126px;overflow:auto;padding-right:2px}.log-item{border-bottom:1px solid var(--line);padding:3px 0;font-size:9.5px;color:#dce5ff;line-height:1.25}.log-item:last-child{border-bottom:none}.modal{position:fixed;inset:0;background:rgba(0,0,0,.78);display:none;align-items:center;justify-content:center;z-index:9999;padding:14px}.modal.open{display:flex}.modal-card{max-width:min(92vw,520px);max-height:92vh;background:#10131a;border:1px solid var(--line);border-radius:16px;padding:10px;box-shadow:0 20px 80px rgba(0,0,0,.5)}.modal-card img{display:block;max-width:100%;max-height:78vh;object-fit:contain;margin:auto}.modal-title{text-align:center;font-weight:850;margin-top:8px;font-size:13px}@media(max-width:640px){.app{padding:4px}.title{font-size:13px}.sub{font-size:9px}.badge{display:none}.player{padding:4px;border-radius:11px}.player-name{font-size:10.5px}.meta-line{font-size:8px}.counts span{min-width:auto;padding:2px 4px;font-size:8px}.counts b{font-size:9px}.field{grid-template-columns:96px minmax(0,1fr);gap:3px}.active-zone{min-height:112px}.card{padding:2px;border-radius:7px;min-height:78px}.card.active{min-height:112px}.thumb{height:50px}.card.active .thumb{height:78px}.card-id{display:none}.card-name{font-size:7px}.card.active .card-name{font-size:8.5px}.hp-row{font-size:6.5px}.energy{font-size:6.5px}.tools{display:none}.bench{gap:2px}.hand-strip{grid-template-columns:repeat(8,minmax(0,1fr));gap:2px}.hand-card{min-height:60px}.center{padding:3px 5px}.big-turn{font-size:15px}.step-line{font-size:8px}.flag{padding:2px 4px;font-size:8px}.bottom{padding:4px}.controls{gap:3px;margin-bottom:4px}button{padding:4px 6px;font-size:10px}.tabs{grid-template-columns:1fr;gap:4px}.panel{padding:4px;min-height:46px}.options,.log{max-height:86px}.option{font-size:9.5px;padding:3px 4px}.log-item{font-size:9px;padding:2px 0}}@media(max-width:390px){.field{grid-template-columns:88px minmax(0,1fr)}.counts span:nth-child(4){display:none}.thumb{height:44px}.card.active .thumb{height:72px}.card{min-height:70px}.card.active{min-height:104px}}
</style></head><body><div class="app"><div class="header"><div><div class="title" id="title"></div><div class="sub" id="subtitle"></div></div><div class="badge" id="result"></div></div><div class="arena"><div class="player p1 opponent" id="player1"></div><div class="center"><div><div class="big-turn" id="turn"></div><div class="step-line" id="stepLine"></div></div><div class="flags"><span class="flag" id="viewer"></span><span class="flag" id="actionCount"></span><span class="flag" id="energyAttached"></span><span class="flag" id="supporterPlayed"></span></div></div><div class="player p0" id="player0"></div></div><div class="bottom"><div class="controls"><button onclick="prevFrame()">←</button><button onclick="togglePlay()" id="playBtn">▶</button><button onclick="nextFrame()">→</button><input id="slider" type="range" min="0" value="0" oninput="goTo(+this.value)"><span class="badge" id="pos"></span></div><div class="tabs"><div class="panel"><div class="panel-title">Options</div><div class="options" id="options"></div></div><div class="panel"><div class="panel-title">Recent logs</div><div class="log" id="logs"></div></div></div></div></div><div class="modal" id="modal" onclick="closeModal()"><div class="modal-card" onclick="event.stopPropagation()"><img id="modalImg"><div class="modal-title" id="modalTitle"></div></div></div><script>
const battle=__BATTLE_DATA__;let idx=0;let timer=null;function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function hpPct(c){if(!c||!c.maxHp)return 0;return Math.max(0,Math.min(100,100*(c.hp??0)/c.maxHp))}function openModal(src,title){if(!src)return;document.getElementById("modalImg").src=src;document.getElementById("modalTitle").textContent=title||"";document.getElementById("modal").classList.add("open")}function closeModal(){document.getElementById("modal").classList.remove("open")}document.addEventListener("keydown",e=>{if(e.key==="Escape")closeModal()});function cardHtml(c,active=false,hand=false){if(!c)return `<div class="card empty"></div>`;const hp=(c.hp!=null&&c.maxHp!=null)?`${c.hp}/${c.maxHp}`:"";const tools=c.tools&&c.tools.length?`<div class="tools">Tool: ${esc(c.tools.join(", "))}</div>`:"";const evo=c.preEvolution&&c.preEvolution.length?`<div class="tools">Evo: ${esc(c.preEvolution.join(" → "))}</div>`:"";const img=c.image?`<img class="thumb" src="${esc(c.image)}" loading="lazy">`:`<div class="thumb"></div>`;const energyNames=c.energyCards&&c.energyCards.length?`<div class="energy-list">${esc(c.energyCards.join(" / "))}</div>`:"";const cls=`card ${active?"active":""} ${hand?"hand-card":""}`;return `<div class="${cls}" onclick="openModal('${esc(c.image||"")}','${esc(c.name)}')">${img}<div class="card-id">#${esc(c.id)} / s${esc(c.serial)}</div><div class="card-name">${esc(c.name)}</div><div class="hp-row"><span>HP</span><b>${esc(hp)}</b></div><div class="hpbar"><div class="hpfill" style="width:${hpPct(c)}%"></div></div><div class="energy">⚡ ${esc(c.energies??0)}${energyNames}</div>${tools}${evo}</div>`}function handHtml(p){if(p.hand===null||p.hand===undefined)return `<div class="hand-hidden">Hand hidden (${esc(p.handCount??"?")} cards)</div>`;if(!p.hand.length)return `<div class="hand-hidden">Hand empty</div>`;return `<div class="hand-strip">${p.hand.map(c=>cardHtml(c,false,true)).join("")}</div>`}function playerHtml(p){const active=p.active&&p.active.length?p.active[0]:null;const bench=[...(p.bench||[])];while(bench.length<5)bench.push(null);const status=Object.entries(p.status||{}).filter(([k,v])=>v).map(([k])=>k).join(", ")||"normal";return `<div class="inner"><div class="player-head"><div><div class="player-name">${esc(p.name)}</div><div class="meta-line">P${p.index} · ${esc(status)}</div></div><div class="counts"><span>H<b>${esc(p.handCount??"?")}</b></span><span>D<b>${esc(p.deckCount??"?")}</b></span><span>P<b>${esc(p.prizeCount??"?")}</b></span><span>X<b>${esc(p.discardCount??"?")}</b></span></div></div><div class="field"><div><div class="zone-title">Active</div><div class="active-zone">${cardHtml(active,true)}</div></div><div><div class="zone-title">Bench</div><div class="bench">${bench.slice(0,5).map(c=>cardHtml(c)).join("")}</div></div></div><div class="zone-title">Hand</div>${handHtml(p)}</div>`}function render(){const f=battle.frames[idx];document.getElementById("title").textContent=battle.title||"Pokémon TCG Battle Viewer";document.getElementById("subtitle").textContent=`Episode ${battle.episodeId??"-"} · ${battle.teamNames.join(" vs ")}`;document.getElementById("result").textContent=`Rewards: ${(battle.rewards||[]).join(" / ")} · Status: ${(battle.statuses||[]).join(" / ")}`;document.getElementById("slider").max=Math.max(0,battle.frames.length-1);document.getElementById("slider").value=idx;document.getElementById("pos").textContent=`${idx+1}/${battle.frames.length}`;document.getElementById("turn").textContent=`TURN ${f.turn??"-"}`;document.getElementById("stepLine").textContent=`Step ${f.step??f.rawStep}`;document.getElementById("viewer").textContent=f.viewer==null?"View -":`View P${f.viewer}`;document.getElementById("actionCount").textContent=`Act ${f.turnActionCount??"-"}`;document.getElementById("energyAttached").textContent=f.energyAttached?"Energy ✓":"Energy -";document.getElementById("supporterPlayed").textContent=f.supporterPlayed?"Supporter ✓":"Supporter -";document.getElementById("player0").innerHTML=playerHtml(f.players[0]||{});document.getElementById("player1").innerHTML=playerHtml(f.players[1]||{});const opts=f.select&&f.select.options?f.select.options:[];document.getElementById("options").innerHTML=opts.length?opts.map((o,i)=>`<div class="option">[${i}] ${esc(o)}</div>`).join(""):`<div class="option">No options</div>`;const logs=f.logs||[];document.getElementById("logs").innerHTML=logs.length?logs.map(x=>`<div class="log-item">${esc(x)}</div>`).join(""):`<div class="log-item">No recent logs</div>`}function goTo(i){idx=Math.max(0,Math.min(battle.frames.length-1,i));render()}function nextFrame(){goTo(idx+1)}function prevFrame(){goTo(idx-1)}function togglePlay(){if(timer){clearInterval(timer);timer=null;document.getElementById("playBtn").textContent="▶"}else{timer=setInterval(()=>{if(idx>=battle.frames.length-1){togglePlay();return}nextFrame()},800);document.getElementById("playBtn").textContent="⏸"}}render();
</script></body></html>'''


def write_html(
    json_path: str | Path,
    out_html: str | Path = "battle_viewer.html",
    card_image_dir: Optional[str | Path] = None,
    card_pdf: Optional[str | Path] = None,
    prepare_images: bool = False,
    overwrite_images: bool = False,
) -> Path:
    out = Path(out_html)
    html_dir = out.parent if out.parent != Path("") else Path(".")

    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    used_card_ids = sorted(_collect_card_ids(raw))

    image_map: Dict[int, str] = {}

    if prepare_images:
        if card_pdf is None:
            raise ValueError("prepare_images=True の場合は card_pdf を指定してください。")
        if card_image_dir is None:
            card_image_dir = html_dir / "card_images"
        prepare_card_images(
            pdf_path=card_pdf,
            out_dir=card_image_dir,
            card_ids=used_card_ids,
            overwrite=overwrite_images,
        )

    if card_image_dir is not None:
        image_map = _existing_card_images(card_image_dir, html_dir=html_dir)

    data = build_battle_data(json_path, image_map=image_map)
    payload = json.dumps(data, ensure_ascii=False)
    html_text = HTML_TEMPLATE.replace("__BATTLE_DATA__", payload)
    out.write_text(html_text, encoding="utf-8")
    return out
