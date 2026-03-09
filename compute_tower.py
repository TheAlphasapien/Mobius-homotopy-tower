"""
compute_tower.py - Compute and save Cayley-Dickson tower levels

Usage:
  python compute_tower.py                    # Compute all levels 16D through 128D
  python compute_tower.py 256               # Compute up to 256D
  python compute_tower.py 64 --from-scratch  # Recompute 64D from scratch
  python compute_tower.py 256 --only         # Compute ONLY 256D (requires 128D saved)

Each level is saved to tower_data/level_XXD/
Subsequent levels use the previous level's multiplication table.

No unicode characters in output (safe for all terminals).

Author: Adam Kevin Morgan / Boundary Paradigm Program
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cd_tower import CDLevel


def compute_and_save(dim, prev_table=None, base_dir="tower_data"):
    """Compute one level and save it."""
    
    def report(msg):
        print(msg, flush=True)
    
    level = CDLevel.compute(dim, prev_table=prev_table, report_fn=report)
    
    save_dir = os.path.join(base_dir, f"level_{dim}D")
    level.save(save_dir)
    level.summary()
    
    return level


def main():
    base_dir = "tower_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # Parse arguments
    max_dim = 128
    only_mode = False
    from_scratch = False
    
    args = sys.argv[1:]
    for arg in args:
        if arg == '--only':
            only_mode = True
        elif arg == '--from-scratch':
            from_scratch = True
        else:
            try:
                max_dim = int(arg)
            except ValueError:
                print(f"Unknown argument: {arg}")
                sys.exit(1)
    
    # Validate
    if max_dim < 16 or (max_dim & (max_dim - 1)) != 0:
        print(f"Dimension must be a power of 2 >= 16, got {max_dim}")
        sys.exit(1)
    
    # Build the list of dimensions to compute
    dims = []
    d = 16
    while d <= max_dim:
        dims.append(d)
        d *= 2
    
    if only_mode:
        dims = [max_dim]
    
    print(f"Tower computation plan: {' -> '.join(str(d) for d in dims)}")
    print(f"Data directory: {os.path.abspath(base_dir)}")
    print()
    
    t_total = time.time()
    prev_table = None
    
    for dim in dims:
        save_dir = os.path.join(base_dir, f"level_{dim}D")
        
        # Check if already computed
        if os.path.exists(os.path.join(save_dir, 'metadata.json')) and not from_scratch:
            print(f"\n--- {dim}D: Already computed, loading ---")
            level = CDLevel.load(save_dir)
            prev_table = level.mult_table
            level.summary()
            continue
        
        # Load previous level's table if available and not from scratch
        if prev_table is None and not from_scratch:
            prev_dim = dim // 2
            prev_dir = os.path.join(base_dir, f"level_{prev_dim}D")
            if os.path.exists(os.path.join(prev_dir, 'mult_table.npy')):
                print(f"\n--- Loading {prev_dim}D table for incremental build ---")
                import numpy as np
                prev_table = np.load(os.path.join(prev_dir, 'mult_table.npy'))
                print(f"  Loaded {prev_dim}x{prev_dim} table")
        
        # Compute
        level = compute_and_save(dim, prev_table=prev_table, base_dir=base_dir)
        prev_table = level.mult_table
    
    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"TOWER COMPLETE")
    print(f"{'='*60}")
    print(f"  Levels: {' -> '.join(str(d) for d in dims)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Data saved in: {os.path.abspath(base_dir)}/")


if __name__ == '__main__':
    main()
