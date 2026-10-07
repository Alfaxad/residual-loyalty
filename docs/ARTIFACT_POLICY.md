# Artifact policy

The repository is a compact public evidence release. It includes artifacts that
allow a reviewer to trace manuscript claims to frozen numerical results without
requiring multi-gigabyte model state.

## Included

- manuscript sources and rendered deliverables;
- aggregate analyses and per-checkpoint decisions;
- raw investigator forecast prompts and responses for the released forecast
  audit;
- frozen configurations, parser contracts, data hashes, and timestamps;
- scientific source code and public release checks;
- derived tables with scripts that regenerate them.

## Excluded

- model adapters and pretrained-model weights;
- activation caches, Jacobian tensors, and other large intermediate arrays;
- account-specific runtime configuration and private infrastructure metadata;
- operational compute and billing ledgers;
- redundant local render scratch files.

The exclusions do not remove the aggregate evidence cited by the manuscript.
They do limit full inference replay without regenerating the large artifacts.
Model and dataset providers retain their own licenses and access requirements.
