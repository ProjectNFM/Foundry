# Job launch snapshots

Hydra local/Slurm launchers create an immutable Git-archive snapshot at
submission time. Queued jobs run from that bundle, not from the live checkout;
no manual worktree is needed.

Before launching jobs:

- Require a clean, committed repository: `git status --short` must be empty.
- Set `FOUNDRY_SNAPSHOT_ROOT` to a compute-node-visible shared directory (use
  `/network/scratch/s/sobralm/foundry-launches` on this cluster).
- Submit Slurm jobs to the `long` partition unless the user explicitly chooses
  another partition.
- Use the normal `python main.py ... -m` workflow. Do not disable snapshots or
  bypass the clean-Git check for production jobs.

After submission, record the Slurm job ID and snapshot bundle path from the
launcher output. 
