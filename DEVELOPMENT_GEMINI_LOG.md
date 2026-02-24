# Gemini Tool Log

## 2026-02-25
- `write_to_file` to log user request for Coins, Relics, and Special Score feature foundations.
- `write_to_file` to draft `task.md` for foundational architecture implementation.
- `write_to_file` to create `implementation_plan.md` outlining the architecture for Coin drops, NPC Shop events, Relic event hooks, and Special Score sequence checks.
- `notify_user` to request approval for the `implementation_plan.md`.
- `view_file` to read `Player` structure and sprite group declarations in `main.py`.
- `multi_replace_file_content` to add `self.coins`, `self.relics`, `self.unlocked_scores` arrays, as well as `Coin` and `ShopNPC` sprite classes and rendering groups.
- `view_file` to inspect `STATE_PLAYING` enemy death blocks in `main.py`.
- `multi_replace_file_content` to implement coin drops on death, Stage-based NPC spawning, and hit-detection collision to trigger `STATE_SHOP`.
- `multi_replace_file_content` to add `STATE_SHOP` UI rendering and HUD `coins` text element.
- `view_file` to locate `active_score` sequence execution logic.
- `multi_replace_file_content` to inject logic that checks the entire array of `active_score` against `unlocked_scores` sequences before iterating individual notes.
- `multi_replace_file_content` to create `trigger_relic_event(event_name, **kwargs)` hook architecture.
- `run_command` `python -m py_compile main.py` to verify syntax.
- `multi_replace_file_content` to update `task.md`.
- `write_to_file` to update `DEVELOPMENT_GEMINI_LOG.md`.
- `multi_replace_file_content` to append foundational changes to `walkthrough.md`.
- `notify_user` to report systems complete.
- `multi_replace_file_content` to hotfix Map Loading JSON parse bug `(too many values to unpack)` by restoring the literal `eval(coord_str)` logic.
- `view_file` to track down map load `NameError: GRID_SIZE`.
- `multi_replace_file_content` to inject `GRID_SIZE = 64` into constants.
