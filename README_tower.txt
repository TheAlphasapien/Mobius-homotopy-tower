CD Tower Computation System
===========================
Author: Adam Kevin Morgan / Boundary Paradigm Program

Three files:

  cd_tower.py        - Core module (algebra, level computation, save/load)
  compute_tower.py   - Build and save tower levels incrementally
  analyze_level.py   - Load saved levels and run analysis tests

Requirements: numpy (no other dependencies)


COMPUTING LEVELS
----------------

  # Compute all levels from 16D through 128D
  python compute_tower.py

  # Compute up to 256D (will take hours for 256D)
  python compute_tower.py 256

  # Recompute a specific level from scratch
  python compute_tower.py 64 --from-scratch

  # Compute ONLY 256D (requires 128D already saved)
  python compute_tower.py 256 --only

Each level is saved to tower_data/level_XXD/ containing:
  - mult_table.npy    (DIM x DIM x DIM float64 - largest file)
  - zd_pairs.npy      (N x 4 int32)
  - edges.npy         (E x 4 int32)
  - metadata.json     (counts, timings)
  - components.json   (per-component structural data)

Previously computed levels are automatically skipped.
Each new level uses the previous level's multiplication
table for incremental construction.


ANALYZING LEVELS
----------------

  # Run all standard tests on a saved level
  python analyze_level.py 64

  # Run specific tests only
  python analyze_level.py 128 --tests overlap chirality

  # Compare two levels
  python analyze_level.py --compare 64 128

  # Summary table of all computed levels
  python analyze_level.py --summary

Available tests:
  summary       Basic counts and distributions
  chirality     Test all component pairs for chiral conjugacy
  overlap       Cross-component kernel overlap (breakpoint detector)
  transparency  Inter-component product norm test
  inheritance   Verify previous level's ZD pairs are preserved
  kernels       Sample kernel dimensions per stratum
  hub_spoke     Analyze mixed-degree component topology


EXPECTED TIMINGS (approximate, single core)
-------------------------------------------

  16D:    < 1 second
  32D:    ~ 2 seconds
  64D:    ~ 30 seconds
  128D:   ~ 15 minutes
  256D:   ~ 4-8 hours (estimated)
  512D:   ~ days (estimated)

The multiplication table precomputation dominates at higher
dimensions. The ZD search scales as O(n^4) in the dimension.


PROGRAMMATIC USE
----------------

  from cd_tower import CDLevel

  # Load a previously computed level
  level = CDLevel.load("tower_data/level_64D")

  # Access the structure
  level.mult_table      # numpy (64, 64, 64)
  level.zd_pairs        # list of (i, j, k, l)
  level.components      # list of sets
  level.comp_info       # list of dicts with metadata
  level.degree_dist     # {degree: count}
  level.undirected_adj  # {node: set of neighbors}

  # Continue to next level
  next_level = CDLevel.compute(128, prev_table=level.mult_table)
  next_level.save("tower_data/level_128D")


NOTES
-----

- No unicode characters in any output (safe for all terminals)
- All file I/O uses numpy .npy format and JSON
- The multiplication table for 128D is ~16MB
- The 256D table would be ~128MB
- The system is single-threaded; parallelizing the ZD search
  would give ~linear speedup across cores
