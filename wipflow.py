#!/usr/bin/env python3
# wipflow.py — append-only, event-sourced WIP limiter for one brilliant human
# No external deps. Python 3.9+ recommended.

import os, shutil, subprocess
import argparse, dataclasses, json, os, sys, time, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from rich.console import Console
from rich.table import Table
from shutil import get_terminal_size
import difflib

console = Console()

ODIN_ROOT_CONFIG = Path.home() / ".config" / "odin" / "root.txt"

with open(ODIN_ROOT_CONFIG, 'r') as f:
    odin_root_str = f.read().strip()
odin_path = Path(odin_root_str).resolve()



APP_DIR = odin_path / "wipflow"
EVENTS = APP_DIR / "events.jsonl"
VERSION = "0.1.0"

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


def _read_ids_from_input(args) -> list[str]:
    """Collect IDs from args or stdin. Accepts whitespace-separated tokens."""
    ids = list(args.ids)
    should_read_stdin = args.stdin or (not ids and not sys.stdin.isatty())
    if should_read_stdin:
        data = sys.stdin.read()
        ids.extend(t for t in data.replace("\0", " ").split() if t.strip())
    # de-dup while preserving order
    seen, out = set(), []
    for t in ids:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

def _resolve_prefix(pid_or_prefix: str, st) -> str | None:
    """Allow unique prefixes. Return full id or None if ambiguous/missing."""
    if pid_or_prefix in st.projects:
        return pid_or_prefix
    matches = [pid for pid in st.projects.keys() if pid.startswith(pid_or_prefix)]
    if len(matches) == 1:
        return matches[0]
    return None  # none or ambiguous

def _score_pair(query: str, candidate: str) -> float:
    """
    Return a similarity score in [0,100]. rapidfuzz if available; else difflib.
    """
    if _HAS_RAPIDFUZZ:
        # token_set_ratio handles word order & duplicates reasonably
        return float(fuzz.token_set_ratio(query, candidate))
    # difflib quick ratio (0..1) -> scale to 0..100
    return 100.0 * difflib.SequenceMatcher(a=query.lower(), b=candidate.lower()).quick_ratio()

def _best_match(query: str, items: Iterable[Tuple[str, str]]) -> Optional[str]:
    """
    items: iterable of (id, title). Return best id or None.
    """
    q = (query or "").strip()
    if not q:
        # empty query: prefer no automatic guess; return None
        return None
    best_id, best_score = None, -1.0
    for pid, title in items:
        s = _score_pair(q, title)
        if s > best_score:
            best_id, best_score = pid, s
    return best_id

# --- interactive prompt (prefer prompt_toolkit; else fallbacks) ---

def _interactive_pick(items: List[Tuple[str, str]], prompt_text: str = "Find project> ") -> Optional[str]:
    """
    items: list of (id, title). Returns selected id or None.
    Tries prompt_toolkit with fuzzy completer; if not available, tries `fzf`;
    finally falls back to a simple numbered picker.
    """
    # 1) prompt_toolkit path
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

        # Show "title  [id]" to allow disambiguation by eye; map back to id
        display_strings = [f"{title}  [{pid}]" for pid, title in items]
        completer = FuzzyCompleter(WordCompleter(display_strings, sentence=True))
        choice = prompt(prompt_text, completer=completer)
        choice = (choice or "").strip()
        if not choice:
            return None
        # exact display-string match → id
        if choice in display_strings:
            idx = display_strings.index(choice)
            return items[idx][0]
        # otherwise fuzzy resolve the entered text against the display set
        bid = _best_match(choice, list(zip([pid for pid,_ in items], display_strings)))
        if bid is not None:
            # _best_match compares against the *display string*, so we got a pid already
            return bid
        # last attempt: fuzzy against titles only
        bid2 = _best_match(choice, items)
        return bid2
    except Exception:
        pass

    # 2) fzf path (external, if installed)
    if shutil.which("fzf"):
        # feed "id \t title" rows into fzf; print id when selected
        lines = [f"{pid}\t{title}" for pid, title in items]
        try:
            proc = subprocess.Popen(
                ["fzf", "--with-nth=2..", "--prompt", prompt_text],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True
            )
            out, _ = proc.communicate("\n".join(lines))
            out = (out or "").strip()
            if not out:
                return None
            # fzf returns the whole selected line; id is first field
            return out.split("\t", 1)[0]
        except Exception:
            pass

    # 3) Minimal numbered fallback (keeps things usable everywhere)
    if not items:
        return None
    print("\nSelect a project:")
    for i, (pid, title) in enumerate(items[:50], 1):
        print(f"{i:2d}. {title}  [{pid}]")
    try:
        sel = input("Number (blank to cancel): ").strip()
        if not sel:
            return None
        idx = int(sel) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
    except Exception:
        return None
    return None

# --- public entry point ---

def find_project_id(
    query: Optional[str],
    *,
    state,                 # your rebuilt State (must provide .projects dict)
    pool: str = "all",     # "queued" | "active" | "incubator" | "done" | "all"
    interactive: bool = False
) -> str:
    """
    Return a single project ID chosen by fuzzy title match.
    - Non-interactive: choose the best match or raise ValueError if none.
    - Interactive: show a fuzzy prompt; return chosen ID or raise ValueError if canceled.
    """
    # collect candidates from pool
    projects = list(state.projects.values())
    if pool != "all":
        projects = [p for p in projects if getattr(p, "state", None) == pool]

    # build (id, title) list
    items: List[Tuple[str, str]] = [(p.id, p.title) for p in projects if getattr(p, "title", None)]

    if interactive:
        pid = _interactive_pick(items)
        if not pid:
            raise ValueError("Selection canceled or no items.")
        return pid

    # non-interactive requires a query
    if not query or not query.strip():
        raise ValueError("Query required when not in interactive mode.")

    pid = _best_match(query, items)
    if not pid:
        raise ValueError("No match found.")
    return pid

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

ProjectState = str  # "incubator","queued","active","completed","parked","killed"
ClassOfService = str  # "gold","silver","bronze"

@dataclass
class Project:
    id: str
    title: str
    state: ProjectState = "incubator"
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    cos: ClassOfService = "silver"   # class of service
    cod: int = 3                     # Cost of Delay (1-5)
    size: int = 3                    # Duration proxy (1-5)
    snss: Optional[str] = None       # Smallest Next Shippable Step
    wildcard: bool = False           # occupies the wildcard slot if active
    notes: Optional[str] = None

@dataclass
class Settings:
    cap: int = 3
    wildcard_slots: int = 1
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

@dataclass
class State:
    settings: Settings = field(default_factory=Settings)
    projects: Dict[str, Project] = field(default_factory=dict)

    @property
    def active(self) -> List[Project]:
        return [p for p in self.projects.values() if p.state == "active"]

    @property
    def active_normal(self) -> List[Project]:
        return [p for p in self.active if not p.wildcard]

    @property
    def active_wildcards(self) -> List[Project]:
        return [p for p in self.active if p.wildcard]

    def tokens_used(self) -> int:
        return len(self.active_normal)

    def tokens_total(self) -> int:
        return self.settings.cap

    def tokens_free(self) -> int:
        return max(0, self.tokens_total() - self.tokens_used())

    def wildcard_used(self) -> int:
        return len(self.active_wildcards)

    def wildcard_total(self) -> int:
        return self.settings.wildcard_slots

    def wildcard_free(self) -> int:
        return max(0, self.wildcard_total() - self.wildcard_used())

# ---------- Rebuilder ----------

def rebuild_state(events: List[dict]) -> State:
    st = State()
    for ev in events:
        k = ev["kind"]
        if k == "settings_initialized":
            st.settings = Settings(cap=ev["cap"], wildcard_slots=ev["wildcard_slots"],
                                   created_at=ev["ts"], updated_at=ev["ts"])
        elif k == "settings_updated":
            st.settings.cap = ev.get("cap", st.settings.cap)
            st.settings.wildcard_slots = ev.get("wildcard_slots", st.settings.wildcard_slots)
            st.settings.updated_at = ev["ts"]
        elif k == "project_created":
            pid = ev["id"]
            st.projects[pid] = Project(
                id=pid, title=ev["title"], state=ev.get("state", "incubator"),
                created_at=ev["ts"], updated_at=ev["ts"],
                cos=ev.get("cos", "silver"), cod=ev.get("cod", 3), size=ev.get("size", 3),
                snss=ev.get("snss"), wildcard=ev.get("wildcard", False),
                notes=ev.get("notes"),
            )
        elif k == "project_updated":
            p = st.projects.get(ev["id"])
            if p:
                for field_name, value in ev.items():
                    if field_name in {"id","kind","ts"}:
                        continue
                    if hasattr(p, field_name):
                        setattr(p, field_name, value)
                p.updated_at = ev["ts"]
        elif k == "project_moved":
            p = st.projects.get(ev["id"])
            if p:
                p.state = ev["to"]
                if "wildcard" in ev:
                    p.wildcard = ev["wildcard"]
                p.updated_at = ev["ts"]
    return st

# ---------- Utilities ----------

def gen_id() -> str:
    return uuid.uuid4().hex[:8]

def print_table(headers, rows, *, max_width=None, wrap_cols=None, align=None):
    """
    headers: tuple/list of column headers
    rows: list of tuples/lists, each same length as headers
    max_width: int | None -> overall table width (defaults to terminal width)
    wrap_cols: set[int] of column indexes to wrap (others will crop)
    align: dict[int,str] -> e.g. {0:'left', 2:'right', 3:'center'}
    """
    if max_width is None:
        max_width = get_terminal_size((120, 20)).columns

    wrap_cols = wrap_cols or set()
    align = align or {}

    table = Table(show_header=True, header_style="bold", expand=True, pad_edge=False)

    # Add columns with per-column overflow behavior
    for i, h in enumerate(headers):
        overflow = "fold" if i in wrap_cols else "crop"
        no_wrap = (i not in wrap_cols)  # crop without wrapping
        justify = {"left":"left","right":"right","center":"center"}.get(align.get(i, "left"), "left")
        table.add_column(str(h), overflow=overflow, no_wrap=no_wrap, justify=justify)

    for r in rows:
        # convert all cells to str (Rich handles ANSI, etc.)
        table.add_row(*[ "" if c is None else str(c) for c in r ])

    console.print(table, width=max_width)

def cos_badge(cos: str) -> str:
    return {"gold":"G","silver":"S","bronze":"B"}.get(cos,"?")

def state_badge(s: str) -> str:
    return {"active":"A","queued":"Q","incubator":"I","completed":"C","parked":"P","killed":"X"}.get(s,"?")

def ratio(p: Project) -> float:
    return (p.cod or 1) / (p.size or 1)

# ---------- Commands ----------

def cmd_init(args):
    if EVENTS.exists():
        print("Already initialized at", APP_DIR)
        return
    append_event("settings_initialized", cap=args.cap, wildcard_slots=args.wildcard)
    print(f"Initialized wipflow at {APP_DIR} (cap={args.cap}, wildcard={args.wildcard})")

def cmd_find(args, st):
    try:
        pid = find_project_id(
            args.query,
            state=st,
            pool=args.pool,
            interactive=args.interactive
        )
        print(pid)  # ID only → perfect for piping
    except ValueError as e:
        # print a friendly message to stderr; non-zero exit for scripts
        print(str(e), file=sys.stderr)
        sys.exit(1)
        
def cmd_settings(args):
    append_event("settings_updated", **({k: getattr(args,k) for k in ["cap","wildcard"] if getattr(args,k) is not None}))
    print("Settings updated.")

def cmd_new(args):
    pid = gen_id()
    append_event("project_created",
                 id=pid, title=args.title, state="incubator" if args.incubator else "queued",
                 cos=args.cos, cod=args.cod, size=args.size, snss=args.snss or None,
                 notes=args.notes or None)
    print(f"Created project {pid}: {args.title} [{ 'incubator' if args.incubator else 'queued' }]")

def cmd_promote(args, st: State):
    p = st.projects.get(args.id)
    if not p:
        sys.exit("No such project.")
    if p.state != "incubator":
        sys.exit("Only incubator projects can be promoted to queued.")
    append_event("project_moved", id=p.id, to="queued")
    print(f"Promoted {p.id} → queued")

def cmd_start(args, st):
    ids_in = _read_ids_from_input(args)
    if not ids_in:
        print("start: no IDs provided (pass IDs as args or use --stdin / pipe).", file=sys.stderr)
        sys.exit(2)

    # local counters so we don’t have to rebuild state after each start
    tokens_free = st.tokens_free()
    wildcard_free = st.wildcard_free()

    started: list[str] = []
    skipped: list[tuple[str, str]] = []  # (id/prefix, reason)

    for token in ids_in:
        resolved = _resolve_prefix(token, st)
        if not resolved:
            skipped.append((token, "unknown or ambiguous id/prefix"))
            continue

        p = st.projects.get(resolved)
        if not p:
            skipped.append((token, "not found"))
            continue

        if p.state not in ("queued", "parked"):
            skipped.append((p.id, f"state is {p.state}, must be queued/parked"))
            continue

        if args.wildcard:
            if wildcard_free <= 0:
                skipped.append((p.id, "no wildcard slot free"))
                continue
            # record event (wildcard doesn’t consume tokens)
            append_event("project_moved", id=p.id, to="active", wildcard=True)
            wildcard_free -= 1
            started.append(p.id)
        else:
            if tokens_free <= 0:
                skipped.append((p.id, "no tokens free"))
                continue
            append_event("project_moved", id=p.id, to="active", wildcard=False)
            tokens_free -= 1
            started.append(p.id)

    # stdout: only the IDs that actually started (pipe-friendly)
    for pid in started:
        print(pid)

    # stderr: reasons for anything that didn’t start
    for pid, reason in skipped:
        print(f"start: skipped {pid}: {reason}", file=sys.stderr)

    if started and not skipped:
        sys.exit(0)
    elif started:
        sys.exit(3)  # partial
    else:
        sys.exit(4)  # none


def cmd_snss(args):
    append_event("project_updated", id=args.id, snss=args.text)
    print(f"SNSS set for {args.id}")

def cmd_update(args):
    payload = {"id": args.id}
    if args.title is not None: payload["title"] = args.title
    if args.cos is not None: payload["cos"] = args.cos
    if args.cod is not None: payload["cod"] = args.cod
    if args.size is not None: payload["size"] = args.size
    if args.notes is not None: payload["notes"] = args.notes
    append_event("project_updated", **payload)
    print(f"Updated {args.id}")

def cmd_move(args, st: State, to: ProjectState):
    p = st.projects.get(args.id)
    if not p:
        sys.exit("No such project.")
    if to == "parked" and p.state == "active":
        append_event("project_moved", id=p.id, to="parked")
        print(f"Parked {p.id}")
    elif to == "completed" and p.state == "active":
        append_event("project_moved", id=p.id, to="completed")
        print(f"Completed {p.id} ✓ (token returned)")
    elif to == "killed":
        append_event("project_moved", id=p.id, to="killed")
        print(f"Killed {p.id} (token returned if active)")
    elif to == "queued" and p.state in ("parked","incubator"):
        append_event("project_moved", id=p.id, to="queued")
        print(f"Queued {p.id}")
    else:
        sys.exit(f"Cannot move from {p.state} to {to}.")

def cmd_list(args, st: State):
    pool = args.pool or "active"
    projs = list(st.projects.values())
    if pool != "all":
        projs = [p for p in projs if p.state == pool]
    # default sort: active first by COS, then cod/size ratio desc
    if pool == "active":
        projs.sort(key=lambda p: ({"gold":0,"silver":1,"bronze":2}.get(p.cos,3), -ratio(p), p.title.lower()))
    else:
        projs.sort(key=lambda p: (p.state, p.title.lower()))
    rows = []
    for p in projs:
        rows.append((p.id, state_badge(p.state), cos_badge(p.cos),
                     f"{p.cod}/{p.size}={ratio(p):.2f}",
                     "★" if (p.wildcard and p.state=="active") else "",
                     p.title,
                     (p.snss or "")[:60]))
    print_table(rows=rows, headers=("id","S","COS","CoD/Size","WC","title","snss"),
                wrap_cols={5, 6})

def cmd_status(st: State):
    print(f"WIP tokens: {st.tokens_used()}/{st.tokens_total()}  (free: {st.tokens_free()})")
    print(f"Wildcard:   {st.wildcard_used()}/{st.wildcard_total()} (free: {st.wildcard_free()})")
    active = st.active
    if active:
        print("\nActive:")
        cmd_list(argparse.Namespace(pool="active"), st)
    else:
        print("\nActive: (none)")

def cmd_suggest(st: State):
    candidates = [p for p in st.projects.values() if p.state == "queued"]
    candidates.sort(key=lambda p: (-ratio(p), {"gold":0,"silver":1,"bronze":2}.get(p.cos,3), p.title.lower()))
    if not candidates:
        print("No queued projects. Promote from incubator or park/finish something.")
        return
    rows = []
    for p in candidates[:10]:
        rows.append((p.id, cos_badge(p.cos), f"{p.cod}/{p.size}={ratio(p):.2f}", p.title))
    print("Top queued by CoD/Size:")
    print_table(rows=rows, headers=("id","COS","score","title"))

def cmd_finish_friday(st: State):
    # Show act-or-park candidates: active with no SNSS or bronze wildcard
    act = st.active
    rows = []
    for p in act:
        flag = []
        if not p.snss: flag.append("no-SNSS")
        if p.cos=="bronze": flag.append("bronze")
        if p.wildcard: flag.append("wildcard")
        rows.append((p.id, cos_badge(p.cos), " ".join(flag), p.title, (p.snss or "")[:60]))
    if rows:
        print("Finish-or-Park candidates:")
        print_table(rows=rows, headers=("id","COS","flags","title","snss"))
    else:
        print("Nothing active to triage. Consider starting from queue.")
    print("\nReminder: return tokens by completing or parking. No new starts on Fridays.")

# ---------- CLI wiring ----------

def main():
    parser = argparse.ArgumentParser(prog="wipflow", description="Append-only WIP limiter (andon, not klaxon).")
    sub = parser.add_subparsers(dest="cmd")

    p_init = sub.add_parser("init", help="Initialize repository")
    p_init.add_argument("--cap", type=int, default=3, help="WIP token cap (normal slots)")
    p_init.add_argument("--wildcard", type=int, default=1, help="Wildcard slots")
    p_init.set_defaults(func=cmd_init)

    p_set = sub.add_parser("settings", help="Update settings")
    p_set.add_argument("--cap", type=int, help="WIP token cap")
    p_set.add_argument("--wildcard", type=int, help="Wildcard slots")
    p_set.set_defaults(func=cmd_settings)

    p_new = sub.add_parser("new", help="Create project")
    p_new.add_argument("title")
    p_new.add_argument("--incubator", action="store_true", help="Start in incubator (default queued)")
    p_new.add_argument("--cos", choices=["gold","silver","bronze"], default="silver")
    p_new.add_argument("--cod", type=int, default=3, choices=range(1,6))
    p_new.add_argument("--size", type=int, default=3, choices=range(1,6))
    p_new.add_argument("--snss", help="Smallest Next Shippable Step")
    p_new.add_argument("--notes", help="Optional notes")
    p_new.set_defaults(func=cmd_new)

    p_prom = sub.add_parser("promote", help="Incubator → queued")
    p_prom.add_argument("id")
    p_prom.set_defaults(func=cmd_promote)

    # --- in your argparse wiring ---
    p_start = sub.add_parser("start", help="Start project(s). IDs as args or from stdin.")
    p_start.add_argument("ids", nargs="*", help="Project IDs (omit to read from stdin)")
    p_start.add_argument("--stdin", action="store_true", help="Read IDs from stdin (newline/space separated)")
    p_start.add_argument("--wildcard", action="store_true", help="Use wildcard slot instead of a token")
    p_start.set_defaults(func=cmd_start)   # ensure your dispatcher calls func(args, st)


    p_snss = sub.add_parser("snss", help="Set SNSS for a project")
    p_snss.add_argument("id")
    p_snss.add_argument("text")
    p_snss.set_defaults(func=cmd_snss)

    p_upd = sub.add_parser("update", help="Update title/cos/cod/size/notes")
    p_upd.add_argument("id")
    p_upd.add_argument("--title")
    p_upd.add_argument("--cos", choices=["gold","silver","bronze"])
    p_upd.add_argument("--cod", type=int, choices=range(1,6))
    p_upd.add_argument("--size", type=int, choices=range(1,6))
    p_upd.add_argument("--notes")
    p_upd.set_defaults(func=cmd_update)

    p_move = sub.add_parser("move", help="Move project to a state")
    p_move.add_argument("id")
    p_move.add_argument("to", choices=["queued","active","completed","parked","killed"])
    p_move.set_defaults(func=lambda a, st: cmd_move(a, st, a.to))

    p_list = sub.add_parser("list", help="List projects in a pool")
    p_list.add_argument("--pool", choices=["active","queued","incubator","completed","parked","killed","all"],
                        default="active")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Show tokens, slots, and current actives")
    p_status.set_defaults(func=lambda a, st: cmd_status(st))

    p_suggest = sub.add_parser("suggest-next", help="Suggest next queued by CoD/Size")
    p_suggest.set_defaults(func=lambda a, st: cmd_suggest(st))

    p_ff = sub.add_parser("finish-friday", help="Triage view for finishing/parking")
    p_ff.set_defaults(func=lambda a, st: cmd_finish_friday(st))

    p_find = sub.add_parser("find", help="Fuzzy-find a project by title; prints ID")
    p_find.add_argument("query", nargs="?", help="Search text (omit if --interactive)")
    p_find.add_argument("--pool", choices=["queued","active","incubator","done","all"], default="all")
    p_find.add_argument("--interactive", "-i", action="store_true", help="Interactive fuzzy prompt")
    p_find.set_defaults(func=lambda a, st: cmd_find(a, st))

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    # Commands that don’t need state:
    if args.cmd in {"init","settings","new","snss","update"}:
        # For update-like commands we still need existence checks sometimes; handled inside.
        if args.cmd == "init":
            args.func(args)
            return
        if args.cmd in {"settings","new"}:
            args.func(args)
            return

    # Rebuild state for most operations
    evs = read_events()
    st = rebuild_state(evs)

    # Commands needing state
    if args.cmd == "promote":
        args.func(args, st)
    elif args.cmd == "start":
        args.func(args, st)
    elif args.cmd == "move":
        args.func(args, st)
    elif args.cmd == "find":
        args.func(args, st)
    elif args.cmd == "list":
        args.func(args, st)
    elif args.cmd == "status":
        args.func(args, st)
    elif args.cmd == "suggest-next":
        args.func(args, st)
    elif args.cmd == "finish-friday":
        args.func(args, st)
    elif args.cmd in {"snss","update"}:
        # These can run without full state, but state isn't harmful either.
        args.func(args)
    else:
        sys.exit("Unknown command.")

if __name__ == "__main__":
    main()
