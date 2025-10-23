#!/usr/bin/env python3
# wipflow.py — tiny, append-only WIP limiter for one brilliant human
# No external deps required (rich optional for pretty tables). Python 3.9+

import argparse, json, sys, time, uuid, os, shutil, subprocess, difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

VERSION = "0.3.1"


# --- Optional niceties ---
try:
    from rich.console import Console
    from rich.table import Table
    from shutil import get_terminal_size
    _HAS_RICH = True
    console = Console()
except Exception:
    _HAS_RICH = False

try:
    from rapidfuzz import fuzz
    _HAS_RAPID = True
except Exception:
    _HAS_RAPID = False


# --- Storage locations ---
CONFIG_PATH = Path.home() / ".config" / "wipflow" / "config.txt"

def _load_config_data_path() -> Path:
    if not CONFIG_PATH.exists():
        sys.exit(f"Config file not found: {CONFIG_PATH}")

    try:
        raw = CONFIG_PATH.read_text(encoding="utf-8")
    except Exception as e:
        sys.exit(f"Failed to read config file {CONFIG_PATH}: {e}")

    cfg = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            # ignore malformed lines silently; could also sys.exit if you prefer strict
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

    data_path_str = cfg.get("data_file_path")
    if not data_path_str:
        sys.exit(f"Missing 'data_file_path' key in config {CONFIG_PATH}")

    # Expand ~ and environment variables, then resolve
    p = Path(os.path.expandvars(os.path.expanduser(data_path_str))).resolve()

    # We don't require the file to exist yet (init may create it), but ensure parent dir is sensible
    parent = p.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            sys.exit(f"Cannot create parent directory for data file {p}: {e}")

    return p

EVENTS = _load_config_data_path()         # full path to events.jsonl (or your chosen file)
APP_DIR = EVENTS.parent   

# ---------- Event Sourcing ----------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def append_event(kind: str, **data):
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as f:
        rec = {"ts": now_iso(), "kind": kind, **data}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def read_events() -> List[dict]:
    if not EVENTS.exists():
        return []
    out = []
    with EVENTS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

# ---------- Domain ----------

ProjectState = str  # "queued","active","completed","killed"

@dataclass
class Project:
    id: str
    title: str
    state: ProjectState = "queued"
    notes: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

@dataclass
class Settings:
    cap: int = 3
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

@dataclass
class State:
    settings: Settings = field(default_factory=Settings)
    projects: Dict[str, Project] = field(default_factory=dict)

    @property
    def active(self) -> List[Project]:
        return [p for p in self.projects.values() if p.state == "active"]

    def tokens_used(self) -> int:
        return len(self.active)

    def tokens_total(self) -> int:
        return self.settings.cap

    def tokens_free(self) -> int:
        return max(0, self.tokens_total() - self.tokens_used())

# ---------- Rebuilder (backward compatible) ----------

def _normalize_state(s: Optional[str]) -> str:
    if not s:
        return "queued"
    s = s.lower()
    if s in ("incubator", "parked"):
        return "queued"
    if s in ("queued","active","completed","killed"):
        return s
    return "queued"

def rebuild_state(events: List[dict]) -> State:
    st = State()
    for ev in events:
        k = ev["kind"]
        if k == "settings_initialized":
            st.settings = Settings(
                cap=ev.get("cap", st.settings.cap),
                created_at=ev["ts"], updated_at=ev["ts"]
            )
        elif k == "settings_updated":
            if "cap" in ev:
                st.settings.cap = ev["cap"]
            st.settings.updated_at = ev["ts"]
        elif k in {"project_created","item_created"}:
            pid = ev["id"]
            title = ev.get("title") or ev.get("name", f"Untitled-{pid}")
            state = _normalize_state(ev.get("state"))
            st.projects[pid] = Project(
                id=pid, title=title, state=state,
                notes=ev.get("notes"),
                created_at=ev["ts"], updated_at=ev["ts"]
            )
        elif k in {"project_updated","item_updated"}:
            p = st.projects.get(ev["id"])
            if p:
                if "title" in ev:      p.title = ev["title"]
                if "notes" in ev:      p.notes = ev["notes"]
                p.updated_at = ev["ts"]
        elif k in {"project_moved","item_moved"}:
            p = st.projects.get(ev["id"])
            if p:
                to = _normalize_state(ev.get("to"))
                p.state = to or p.state
                p.updated_at = ev["ts"]
    return st

# ---------- Matching / Finder ----------

def _score_pair(query: str, candidate: str) -> float:
    if _HAS_RAPID:
        return float(fuzz.token_set_ratio(query, candidate))
    return 100.0 * difflib.SequenceMatcher(a=query.lower(), b=candidate.lower()).ratio()

def _best_match(query: str, items: Iterable[Tuple[str, str]]) -> Optional[str]:
    q = (query or "").strip()
    if not q:
        return None
    best_id, best_score = None, -1.0
    for pid, title in items:
        s = _score_pair(q, title)
        if s > best_score:
            best_id, best_score = pid, s
    return best_id

def _interactive_pick(items: List[Tuple[str, str]], prompt_text: str = "Find item> ") -> Optional[str]:
    # 1) prompt_toolkit
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
        display = [f"{title}  [{pid}]" for pid, title in items]
        comp = FuzzyCompleter(WordCompleter(display, sentence=True))
        choice = (prompt(prompt_text, completer=comp) or "").strip()
        if not choice:
            return None
        if choice in display:
            return items[display.index(choice)][0]
        return _best_match(choice, items)
    except Exception:
        pass
    # 2) fzf (external)
    if shutil.which("fzf"):
        lines = [f"{pid}\t{title}" for pid, title in items]
        try:
            proc = subprocess.Popen(
                ["fzf", "--with-nth=2..", "--prompt", prompt_text],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            out, _ = proc.communicate("\n".join(lines))
            out = (out or "").strip()
            if not out:
                return None
            return out.split("\t", 1)[0]
        except Exception:
            pass
    # 3) basic numbered fallback
    if not items:
        return None
    for i, (pid, title) in enumerate(items[:50], 1):
        print(f"{i:2d}. {title}  [{pid}]")
    try:
        sel = input("Number (blank to cancel): ").strip()
        if not sel: return None
        idx = int(sel) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
    except Exception:
        return None
    return None

def find_project_id(query: Optional[str], *, state, pool: str = "all", interactive: bool = False) -> str:
    projects = list(state.projects.values())
    if pool != "all":
        projects = [p for p in projects if p.state == pool]
    items: List[Tuple[str, str]] = [(p.id, p.title) for p in projects if p.title]
    if interactive:
        pid = _interactive_pick(items)
        if not pid:
            raise ValueError("Selection canceled or no items.")
        return pid
    if not query or not query.strip():
        raise ValueError("Query required when not in interactive mode.")
    pid = _best_match(query, items)
    if not pid:
        raise ValueError("No match found.")
    return pid

# ---------- Utilities ----------

def gen_id() -> str:
    return uuid.uuid4().hex[:8]

def _print_table(headers, rows):
    if _HAS_RICH:
        try:
            from shutil import get_terminal_size
            width = get_terminal_size((120, 20)).columns
        except Exception:
            width = None
        table = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)
        for h in headers:
            table.add_column(str(h), overflow="fold", no_wrap=False, justify="left")
        for r in rows:
            table.add_row(*[ "" if c is None else str(c) for c in r ])
        console.print(table, width=width)
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join("" if c is None else str(c) for c in r))

def state_badge(s: str) -> str:
    return {"active":"A","queued":"Q","completed":"C","killed":"X"}.get(s,"?")

def _read_ids_from_input(args) -> List[str]:
    ids = list(getattr(args, "ids", []) or [])
    should_read_stdin = getattr(args, "stdin", False) or (not ids and not sys.stdin.isatty())
    if should_read_stdin:
        data = sys.stdin.read()
        ids.extend(t for t in data.replace("\0", " ").split() if t.strip())
    seen, out = set(), []
    for t in ids:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _resolve_prefix(pid_or_prefix: str, st) -> Optional[str]:
    if pid_or_prefix in st.projects:
        return pid_or_prefix
    matches = [pid for pid in st.projects.keys() if pid.startswith(pid_or_prefix)]
    if len(matches) == 1:
        return matches[0]
    return None

# ---------- Commands ----------

def cmd_init(args):
    if EVENTS.exists():
        print("Already initialized at", APP_DIR); return
    append_event("settings_initialized", cap=args.cap)
    print(f"Initialized wipflow at {APP_DIR} (cap={args.cap})")

def cmd_settings(args):
    append_event("settings_updated", **({k: getattr(args,k) for k in ["cap"] if getattr(args,k) is not None}))
    print("Settings updated.")

def cmd_new(args):
    pid = gen_id()
    append_event("item_created", id=pid, title=args.title, state="queued", notes=args.notes or None)
    print(f"Created {pid}: {args.title} [queued]")

def cmd_start(args, st):
    ids_in = _read_ids_from_input(args)
    if not ids_in:
        print("start: no IDs provided (pass IDs as args or use --stdin / pipe).", file=sys.stderr)
        sys.exit(2)
    tokens_free = st.tokens_free()
    started, skipped = [], []
    for token in ids_in:
        resolved = _resolve_prefix(token, st)
        if not resolved:
            skipped.append((token, "unknown or ambiguous id/prefix")); continue
        p = st.projects.get(resolved)
        if not p:
            skipped.append((token, "not found")); continue
        if p.state != "queued":
            skipped.append((p.id, f"state is {p.state}, must be queued")); continue
        if tokens_free <= 0:
            skipped.append((p.id, "no tokens free")); continue
        append_event("item_moved", id=p.id, to="active")
        tokens_free -= 1
        started.append(p.id)
    for pid in started: print(pid)
    for pid, reason in skipped: print(f"start: skipped {pid}: {reason}", file=sys.stderr)
    sys.exit(0 if started and not skipped else (3 if started else 4))

def cmd_move(args, st, to: ProjectState):
    p = st.projects.get(args.id)
    if not p:
        sys.exit("No such item.")

    if to == "queued":
        if p.state == "active":
            append_event("item_moved", id=p.id, to="queued")
            print(f"Paused {p.id} → queued (token returned)")
        else:
            sys.exit(f"Cannot move {p.id} from {p.state} to queued. Use 'start' for queued→active.")

    elif to == "completed":
        if p.state == "active":
            append_event("item_moved", id=p.id, to="completed")
            print(f"Completed {p.id} ✓ (token returned)")
        else:
            sys.exit(f"Only active items can be completed (current state: {p.state}).")

    elif to == "killed":
        # ✅ allow killing from active OR queued
        if p.state in ("active", "queued"):
            append_event("item_moved", id=p.id, to="killed")
            if p.state == "active":
                print(f"Killed {p.id} (token returned)")
            else:
                print(f"Killed {p.id} (was queued; no token involved)")
        else:
            sys.exit(f"Only active or queued items can be killed (current state: {p.state}).")

def cmd_reopen(args, st):
    """Reverse accidental completed/killed. Back to queued by default, or --to active if token free."""
    p = st.projects.get(args.id)
    if not p:
        sys.exit("No such item.")
    if p.state not in ("completed", "killed"):
        sys.exit(f"Item {p.id} is {p.state}; only completed/killed can be reopened.")
    target = args.to or "queued"
    if target == "active":
        if st.tokens_free() <= 0:
            sys.exit("No tokens free to reopen directly to active.")
        append_event("item_moved", id=p.id, to="active")
        print(f"Reopened {p.id} → active")
    elif target == "queued":
        append_event("item_moved", id=p.id, to="queued")
        print(f"Reopened {p.id} → queued")
    else:
        sys.exit("Invalid --to. Use queued (default) or active.")

def cmd_update(args):
    payload = {"id": args.id}
    if args.title is not None: payload["title"] = args.title
    if args.notes is not None: payload["notes"] = args.notes
    if len(payload) == 1:
        print("Nothing to update."); return
    append_event("item_updated", **payload)
    print(f"Updated {args.id}")

def cmd_list(args, st: State):
    pool = args.pool or "active"
    projs = list(st.projects.values())
    if pool != "all":
        projs = [p for p in projs if p.state == pool]
    if pool == "active":
        projs.sort(key=lambda p: p.title.lower())
    else:
        projs.sort(key=lambda p: (p.state, p.title.lower()))
    rows = [(p.id, state_badge(p.state), p.title) for p in projs]
    _print_table(headers=("id","S","title"), rows=rows)

def cmd_status(st: State):
    print(f"WIP tokens: {st.tokens_used()}/{st.tokens_total()}  (free: {st.tokens_free()})")
    act = [ (p.id, p.title) for p in st.active ]
    if act:
        print("\nActive:")
        _print_table(("id","title"), act)
    else:
        print("\nActive: (none)")

def cmd_find(args, st):
    try:
        pid = find_project_id(
            args.query,
            state=st,
            pool=args.pool,
            interactive=args.interactive
        )
        print(pid)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

# ---------- CLI wiring ----------

def main():
    parser = argparse.ArgumentParser(prog="wipflow", description="Append-only WIP limiter.")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Initialize repository")
    p_init.add_argument("--cap", type=int, default=3, help="WIP token cap")
    p_init.set_defaults(func=cmd_init)

    p_set = sub.add_parser("settings", help="Update settings")
    p_set.add_argument("--cap", type=int, help="WIP token cap")
    p_set.set_defaults(func=cmd_settings)

    p_new = sub.add_parser("new", help="Create item (queued)")
    p_new.add_argument("title")
    p_new.add_argument("--notes", help="Optional notes")
    p_new.set_defaults(func=cmd_new)

    p_start = sub.add_parser("start", help="Start item(s). IDs as args or from stdin.")
    p_start.add_argument("ids", nargs="*", help="Item IDs (omit to read from stdin)")
    p_start.add_argument("--stdin", action="store_true", help="Read IDs from stdin (newline/space separated)")
    p_start.set_defaults(func=cmd_start)

    p_move = sub.add_parser("move", help="Move active item to queued/completed/killed")
    p_move.add_argument("id")
    p_move.add_argument("to", choices=["queued","completed","killed"])
    p_move.set_defaults(func=lambda a, st: cmd_move(a, st, a.to))

    p_reopen = sub.add_parser("reopen", help="Reverse completed/killed → queued (default) or --to active")
    p_reopen.add_argument("id")
    p_reopen.add_argument("--to", choices=["queued","active"])
    p_reopen.set_defaults(func=cmd_reopen)

    p_upd = sub.add_parser("update", help="Update title/notes")
    p_upd.add_argument("id")
    p_upd.add_argument("--title")
    p_upd.add_argument("--notes")
    p_upd.set_defaults(func=cmd_update)

    p_list = sub.add_parser("list", help="List items in a pool (default: active)")
    p_list.add_argument("pool", nargs="?", choices=["active","queued","completed","killed","all"], default="active")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Show tokens and current actives")
    p_status.set_defaults(func=lambda a, st: cmd_status(st))

    p_find = sub.add_parser("find", help="Fuzzy-find an item by title; prints ID (pipe-friendly)")
    p_find.add_argument("query", nargs="?", help="Search text (omit if --interactive)")
    p_find.add_argument("--pool", choices=["queued","active","completed","killed","all"], default="all")
    p_find.add_argument("--interactive", "-i", action="store_true", help="Interactive fuzzy prompt")
    p_find.set_defaults(func=cmd_find)

    # complete: alias for move <id> completed
    p_complete = sub.add_parser("complete", help="Mark active item as completed (token returns)")
    p_complete.add_argument("id")
    p_complete.set_defaults(func=lambda a, st: cmd_move(a, st, "completed"))

    # kill: alias for move <id> killed
    p_kill = sub.add_parser("kill", help="Mark item as killed (from active or queued)")
    p_kill.add_argument("id")
    p_kill.set_defaults(func=lambda a, st: cmd_move(a, st, "killed"))

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help(); sys.exit(0)

    # Commands that don’t need prior state (they still append events)
    if args.cmd in {"init","settings","new","update"}:
        args.func(args); return

    # Load state
    evs = read_events()
    st = rebuild_state(evs)

    # Dispatch with state
    if args.cmd == "start":
        args.func(args, st)
    elif args.cmd == "move":
        args.func(args, st)
    elif args.cmd == "complete":
        args.func(args, st)
    elif args.cmd == "kill":
        args.func(args, st)
    elif args.cmd == "reopen":
        args.func(args, st)
    elif args.cmd == "list":
        args.func(args, st)
    elif args.cmd == "status":
        args.func(args, st)
    elif args.cmd == "find":
        args.func(args, st)
    else:
        sys.exit("Unknown command.")

if __name__ == "__main__":
    main()
