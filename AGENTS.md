<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 80
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/mnt/data/LiuXiao/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/mnt/data/LiuXiao/VAS/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /mnt/data/LiuXiao/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/mnt/data/LiuXiao/VAS" --reconcile`
For copied Codex installs, use:
`bash /mnt/data/LiuXiao/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/mnt/data/LiuXiao/VAS"`
<!-- ARIS-CODEX:END -->
