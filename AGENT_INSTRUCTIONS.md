## Goal
Your aim is to make as much progress as possible against ./migration_plan/MIGRATION_PLAN.md.

## Operating rules
- You are running fully autonomously while I sleep. NEVER stop to ask me anything.
  When uncertain, make the most reasonable decision, write a one-line note about it
  in NOTES_FOR_CLIFF.md, and keep going.
- If one task is blocked, do not stop — pick the next available task. Only idle if
  literally everything is blocked, and even then leave a clear note.
- For substantial sub-tasks, decompose and use subagents/workflows; be thorough.

## Commit rules
- Commit after each logical, *working* increment — not one giant commit at the end, but not dozens of commits per milestone either.
- Before committing: run the tests/build if they exist; only commit if green
  (if you must commit red, say so in the message).
- Use clear, conventional commit messages. Keep the default `Co-Authored-By: Claude`
  trailer — I want commits attributed to both of us.
- Commit to the current branch only. NEVER commit on another branch or push to GitHub.

## Safety (best judgment, but never do these unattended)
- No force-push, no history rewrites, no deleting branches.
- No `rm -rf` outside the repo, no touching anything outside this working tree (however, you CAN read from other folders in the home folder, especially ~/starsim and ~/hpvsim).
- No external side effects (no emails, no posting, no deploys, no prod credentials).
- Do NOT `git push` -- nothing should leave this machine.
