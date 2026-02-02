# Economic Dispatch (ED) Solver - Debugging Repository

This repository is for debugging a PuLP formulation of the Economic Dispatch problem that has been coded following the formulation documented in [UnitCommitment.jl](https://anl-ceeesa.github.io/UnitCommitment.jl/0.4/guides/problem/).

## Problem Statement

The code implements an Economic Dispatch solver using PuLP with CBC. It takes a fixed unit commitment schedule (from Julia's MILP solution) and optimizes the power dispatch to minimize operational costs. The formulation follows the UC+ED problem structure from UnitCommitment.jl, but solves only the ED part with fixed commitment decisions.

## Current Issue

**The results should be identical (or extremely close) to Julia's solution**, since this solver takes the exact same commitment decisions from Julia and only optimizes the dispatch. However:

- Most instances show small differences (typically under 0.03%)
- Some instances show larger differences (e.g., instance_2027_Q3_59 shows -0.1591%)
- **These differences should NOT occur** - they indicate a bug in the formulation

**I'm probably missing some constraints or have an error in the implementation.** The PuLP formulation is supposed to match the Economic Dispatch component from the [UnitCommitment.jl formulation](https://anl-ceeesa.github.io/UnitCommitment.jl/0.4/guides/problem/), but something is not quite right.

**Help needed:** If you can spot what's wrong with the formulation or what constraints might be missing, please open an issue or submit a PR!

## Files Required

- `ed_pulp.py` - Main solver script
- `instance_*/InputData.json.gz` - Problem parameters for each instance
- `instance_*/OutputData.json.gz` - Julia's MILP solution (commitment schedule) for each instance

## Installation

```bash
pip install pulp numpy
```

## Usage

```bash
python ed_pulp.py
```

The script will automatically detect and run all instances in subdirectories matching `instance_*` pattern.

## Example Output

```
Found 11 instances

================================================================================

Processing: instance_2021_Q1_1
  PuLP (LP):     $13,830,582.17
  Julia (MILP):  $13,834,078.54
  Difference:    -0.0253%

...

Processing: instance_2027_Q3_59
  PuLP (LP):     $15,227,254.51
  Julia (MILP):  $15,251,517.99
  Difference:    -0.1591%

================================================================================

SUMMARY OF ALL INSTANCES
================================================================================
Instance                             PuLP Cost      Julia Cost     Diff %
--------------------------------------------------------------------------------
instance_2021_Q1_1             $ 13,830,582.17 $ 13,834,078.54   -0.0253%
...
instance_2027_Q3_59            $ 15,227,254.51 $ 15,251,517.99   -0.1591%
--------------------------------------------------------------------------------
Average difference:                                              -0.0220%
```

## Model Formulation

The solver implements:
- **Piecewise linear production costs** for thermal generators
- **Ramping constraints** (up/down, startup, shutdown limits)
- **Storage dynamics** with charge/discharge efficiency
- **Profiled generation** (wind, solar, hydro with time-varying capacity)
- **Load shedding** with penalty costs
- **Power balance** across all time periods

The formulation follows [UnitCommitment.jl's problem specification](https://anl-ceeesa.github.io/UnitCommitment.jl/0.4/guides/problem/), but there appears to be an error or missing constraint somewhere.

## Contributing

**Help wanted!** If you can identify what's wrong with the formulation, please:
- Open an issue explaining the problem
- Submit a PR with the fix
- Point out missing constraints or incorrect implementations

Any help debugging this formulation would be greatly appreciated!

## License

See LICENSE file for details.
