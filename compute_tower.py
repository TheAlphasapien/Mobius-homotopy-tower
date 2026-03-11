"""
compute_tower.py — Compute and save Cayley–Dickson tower levels

Now runs in **two phases per level** so you can always resume:
  Phase 1: build & SAVE mult_table.npy immediately
  Phase 2: ZD search with checkpoint/resume, then graph & analysis

Usage examples:
  python compute_tower.py 512 --min-dim 1 --zd-checkpoint --zd-resume --zd-workers 4
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cd_tower import CDLevel, compute_mult_table, find_zero_divisors, build_graph_and_components, analyze_components
import numpy as np

def is_pow2(x:int)->bool:
    return x>0 and (x & (x-1))==0

def parse_args(argv):
    p = argparse.ArgumentParser(description="Compute Cayley–Dickson tower levels (two-phase save)")
    p.add_argument("max_dim", nargs="?", type=int, default=128,
                   help="Power-of-two dimension ≥ 1 (default: 128)")
    p.add_argument("--min-dim", dest="min_dim", type=int, default=16,
                   help="Lowest power-of-two to compute (default: 16; use 1 to build from base)")
    p.add_argument("--only", action="store_true",
                   help="Compute ONLY max_dim (ignores --min-dim)")
    p.add_argument("--from-scratch", action="store_true",
                   help="Recompute levels ignoring cached data")
    p.add_argument("--workers", type=int, default=None,
                   help="Workers for mult-table build (default: auto)")
    p.add_argument("--zd-workers", type=int, default=None,
                   help="Workers for zero-divisor search (default: auto)")
    p.add_argument("--zd-checkpoint", action="store_true",
                   help="Enable checkpoint/resume for ZD (writes block files & progress)")
    p.add_argument("--zd-resume", action="store_true",
                   help="Resume ZD from prior checkpoint")
    p.add_argument("--prune-cheap", action="store_true",
                   help="Enable safe cheap pruning bound (no false negatives)")
    p.add_argument("--eps", type=float, default=1e-10,
                   help="Squared-norm threshold for ZD test (default: 1e-10)")
    return p.parse_args(argv)

def save_status(save_dir, phase, timings=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'status.json')
    data = {'phase': phase, 'time': time.time()}
    if timings:
        data['timings'] = timings
    with open(path+'.tmp','w') as f:
        json.dump(data, f, indent=2)
    os.replace(path+'.tmp', path)

def main():
    base_dir = "tower_data"; os.makedirs(base_dir, exist_ok=True)
    a = parse_args(sys.argv[1:])
    if not is_pow2(a.max_dim) or not is_pow2(a.min_dim) or a.min_dim>a.max_dim:
        print("Dimensions must be powers of two and min≤max."); sys.exit(1)
    dims = [a.max_dim] if a.only else [d for d in [1<<k for k in range(0, 40)] if d>=a.min_dim and d<=a.max_dim]
    print(f"Plan: {' -> '.join(map(str,dims))}")
    prev_table = None
    t_total = time.time()

    for dim in dims:
        save_dir = os.path.join(base_dir, f"level_{dim}D")
        meta_path = os.path.join(save_dir, 'metadata.json')
        mt_path = os.path.join(save_dir, 'mult_table.npy')

        # Load cached level if allowed
        if os.path.exists(meta_path) and not a.from_scratch:
            print(f"\n--- {dim}D: Already computed, loading ---")
            lvl = CDLevel.load(save_dir)
            prev_table = lvl.mult_table
            lvl.summary(); continue

        # Try increment from previous on disk when not from scratch
        if prev_table is None and not a.from_scratch and dim>1:
            pdir = os.path.join(base_dir, f"level_{dim//2}D")
            pmt = os.path.join(pdir, 'mult_table.npy')
            if os.path.exists(pmt):
                print(f"\n--- Loading {dim//2}D table for incremental build ---")
                prev_table = np.load(pmt, mmap_mode='r')

        # PHASE 1: build table (or reuse if already saved)
        t0 = time.time()
        if os.path.exists(mt_path) and not a.from_scratch:
            print(f"\n[{dim}D] Using existing mult_table.npy (resume)")
            M = np.load(mt_path, mmap_mode='r')
            t_build = 0.0
        else:
            print(f"\n[{dim}D] Step 1: Multiplication table (two-phase save)")
            M = compute_mult_table(dim, prev_table, report_fn=lambda m: print(m, flush=True), workers=a.workers)
            t_build = time.time()-t0
            os.makedirs(save_dir, exist_ok=True)
            np.save(mt_path, M)
            save_status(save_dir, phase='table_done', timings={'mult_table': t_build})
            print(f"[{dim}D] mult_table.npy saved ({t_build:.1f}s)")

        # PHASE 2: ZD, graph, analysis (with checkpoint)
        print(f"[{dim}D] Step 2: Zero divisor search (checkpointable)")
        t1 = time.time()
        zd_pairs = find_zero_divisors(M, dim, report_fn=lambda m: print(m, flush=True),
                                      workers=a.zd_workers, eps=a.eps,
                                      checkpoint_dir=save_dir if a.zd_checkpoint else None,
                                      resume=a.zd_resume, prune=a.prune_cheap)
        t_zd = time.time()-t1
        print(f"[{dim}D] ZD done: {len(zd_pairs)} pairs ({t_zd:.1f}s)")

        print(f"[{dim}D] Step 3: Graph & components")
        t2 = time.time()
        undirected_adj, components = build_graph_and_components(zd_pairs)
        t_graph = time.time()-t2

        print(f"[{dim}D] Step 4: Component analysis")
        t3 = time.time()
        comp_info = analyze_components(components, undirected_adj, dim)
        t_ana = time.time()-t3

        # Save final via CDLevel for compatibility
        lvl = CDLevel(); lvl.dim = dim
        lvl.mult_table = np.load(mt_path) if isinstance(M, np.memmap) else M
        lvl.zd_pairs = zd_pairs
        lvl.undirected_adj = undirected_adj
        lvl.components = components
        lvl.comp_info = comp_info
        # Build degree dist
        from collections import defaultdict
        dd = defaultdict(int)
        for node, nbs in undirected_adj.items():
            if node[0]=='L': dd[len(nbs)] += 1
        lvl.degree_dist = dict(sorted(dd.items()))
        lvl.timings = {'mult_table': float(t_build), 'zd_search': float(t_zd), 'graph': float(t_graph), 'analysis': float(t_ana)}
        lvl.save(save_dir)
        save_status(save_dir, phase='complete', timings=lvl.timings)
        lvl.summary()
        prev_table = lvl.mult_table

    T = time.time()-t_total
    print("\n"+"="*60) ; print("TOWER COMPLETE"); print("="*60)
    print(f" Levels: {' -> '.join(map(str,dims))}")
    print(f" Total time: {T:.1f}s ({T/60:.1f} min)")
    print(f" Data saved in: {os.path.abspath(base_dir)}/")

if __name__=='__main__':
    main()
