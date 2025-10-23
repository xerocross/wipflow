# wipflow

**wipflow** is a tiny, append-only command-line tool for managing your personal Work-In-Progress (WIP).
It helps you stay focused by enforcing a strict limit on how many things you’re working on at once.

No databases, no daemons, no sync conflicts. Just a local log of events and a few commands that keep you honest.

---

## 🧭 Philosophy

* **Focus beats motion.** WIP limits aren’t bureaucracy; they’re self-defense.
* **Everything append-only.** You never lose history — every change is an event.
* **No automation, no guessing.** The tool won’t “suggest” what’s next. You decide.
* **Fast, local, simple.** It works entirely offline, plays well with `git`, `syncthing`, or `rsync`.

---

## ⚙️ Installation

```bash
# Put wipflow somewhere on your PATH
chmod +x wipflow.py
mv wipflow.py ~/bin/wipflow
```

Requirements:

* Python 3.9+
* (optional) [`rich`](https://pypi.org/project/rich/) for pretty tables
* (optional) [`rapidfuzz`](https://pypi.org/project/rapidfuzz/) for faster fuzzy matching
* (optional) [`prompt_toolkit`](https://pypi.org/project/prompt_toolkit/) or [`fzf`](https://github.com/junegunn/fzf) for interactive picking

---

## 🪄 Quick Start

```bash
wipflow init --cap 3        # limit yourself to 3 active items
wipflow new "Write blog post on focus"
wipflow new "Rewire backup scripts"
wipflow start <id>          # move from queued → active (consumes a token)
wipflow complete <id>       # active → completed (token returns)
wipflow kill <id>           # active → killed (token returns)
```

Each change is written to `~/.config/odin/root.txt` → `<odin_root>/wipflow/events.jsonl`.

---

## 🧩 States

| State         | Meaning                              | Token  |
| ------------- | ------------------------------------ | ------ |
| **queued**    | In backlog, not started              | —      |
| **active**    | Currently being worked on            | uses 1 |
| **completed** | Done, finished, success              | freed  |
| **killed**    | Explicitly dropped / no longer doing | freed  |

You can always reverse a mistake:

```bash
wipflow reopen <id>              # completed/killed → queued
wipflow reopen <id> --to active  # if a token is free
```

---

## 🧰 Commands

| Command                                 | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `init --cap N`                          | Initialize repository with WIP cap            |
| `new "title"`                           | Add new queued item                           |
| `start ID...`                           | Start one or more queued items (respects cap) |
| `complete ID`                           | Mark active item completed                    |
| `kill ID`                               | Mark active item killed                       |
| `move ID queued`                        | Pause active item and return token            |
| `reopen ID [--to active]`               | Undo completed/killed                         |
| `update ID [--title ...] [--notes ...]` | Edit metadata                                 |
| `list [pool]`                           | Show items in a pool (`active` default)       |
| `status`                                | Show tokens and current actives               |
| `find [query] [-i]`                     | Fuzzy-find item ID (pipe-friendly)            |
| `settings --cap N`                      | Change WIP token cap                          |

Examples:

```bash
wipflow list queued
wipflow find -i | wipflow start --stdin
```

---

## 🧩 Tips

* **Combine it with other tools.** `wipflow find -i | wipflow start --stdin` lets you fuzzy-search and start in one step.
* **Keep it in version control.** The `events.jsonl` file is append-only; it’s safe to `git commit` or sync.
* **Share across machines.** Works seamlessly with `Syncthing` or `rsync`; no database locking.

---

## 🪶 Design Notes

* The only mutable thing is your *intent*; everything else is an event log.
* The CLI is stateless — it rebuilds current state from the event stream each time.
* You can inspect or visualize history with any JSON-capable tool.

---

## 🧑‍💻 Example Workflow

```bash
wipflow new "Fix mail parser"
wipflow new "Add 'complete' alias to CLI"
wipflow list queued
wipflow start f6c91b2
wipflow status
# ... later ...
wipflow complete f6c91b2
```

---

## 🧾 License

MIT License — use it, fork it, modify it, keep it simple.

---

## ❤️ Acknowledgements

Born from conversations about personal focus and sustainable productivity.
Built for humans who want tools that *assist* instead of *interfere*.
