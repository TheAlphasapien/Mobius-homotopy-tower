"""
analyze_level.py - Load saved tower level and run analysis tests

Usage:
  python analyze_level.py 64                    # Run all standard tests on 64D
  python analyze_level.py 128 --tests overlap chirality  # Specific tests only
  python analyze_level.py 128 --compare 64      # Compare two levels
  python analyze_level.py all --summary         # Summary table of all levels

Available tests:
  summary       - Basic counts and distributions
  chirality     - Test all component pairs for chiral conjugacy
  overlap       - Cross-component kernel overlap (breakpoint detector)
  transparency  - Inter-component product norm test
  inheritance   - Verify previous level's ZD pairs are preserved
  kernels       - Sample kernel dimensions per stratum
  fano          - Check Fano plane connections
  hub_spoke     - Analyze mixed-degree component topology
  
No unicode characters in output.

Author: Adam Kevin Morgan / Boundary Paradigm Program
"""

import sys
import os
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cd_tower import CDLevel, CayleyDickson


BASE_DIR = "tower_data"


def load_level(dim):
    """Load a computed level."""
    save_dir = os.path.join(BASE_DIR, f"level_{dim}D")
    if not os.path.exists(save_dir):
        print(f"Level {dim}D not found in {BASE_DIR}/")
        print(f"Run: python compute_tower.py {dim}")
        sys.exit(1)
    return CDLevel.load(save_dir)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_summary(level):
    """Basic summary statistics."""
    level.summary()


def test_chirality(level):
    """Find all chiral conjugate pairs among components."""
    print(f"\n--- CHIRALITY TEST ({level.dim}D) ---")
    
    chiral_pairs = []
    for i in range(len(level.comp_info)):
        for j in range(i+1, len(level.comp_info)):
            ci = level.comp_info[i]
            cj = level.comp_info[j]
            if ci['left_pairs'] == cj['right_pairs'] and ci['right_pairs'] == cj['left_pairs']:
                chiral_pairs.append((i, j))
    
    paired = set()
    for i, j in chiral_pairs:
        paired.add(i)
        paired.add(j)
    unpaired = [i for i in range(len(level.comp_info)) if i not in paired]
    
    print(f"  Chiral pairs found: {len(chiral_pairs)}")
    print(f"  Self-adjoint components: {len(unpaired)}")
    
    # Group by component type
    type_chirality = defaultdict(lambda: {'chiral': 0, 'self_adj': 0})
    for i, j in chiral_pairs:
        sig = level.comp_info[i]['sig']
        type_chirality[sig]['chiral'] += 1
    for i in unpaired:
        sig = level.comp_info[i]['sig']
        type_chirality[sig]['self_adj'] += 1
    
    print(f"\n  By component type:")
    for sig, counts in sorted(type_chirality.items(), key=lambda x: -x[0][0]):
        size, degs = sig
        print(f"    size={size} degs={dict(degs)}: "
              f"{counts['chiral']} pairs + {counts['self_adj']} self-adj")
    
    return chiral_pairs, unpaired


def test_overlap(level, n_sample=10):
    """Cross-component kernel overlap test (breakpoint detector)."""
    print(f"\n--- KERNEL OVERLAP TEST ({level.dim}D) ---")
    
    A = CayleyDickson(level.dim)
    basis = np.eye(level.dim)
    max_deg = max(level.degree_dist.keys())
    
    print(f"  Highest degree: {max_deg}")
    
    # Find components with max-degree nodes
    candidates = []
    for ci_info in level.comp_info:
        if max_deg in ci_info['deg_counts']:
            # Find a max-degree node in this component
            comp = level.components[ci_info['ci']]
            for node in sorted(comp):
                if node[0] == 'L' and len(level.undirected_adj[node]) == max_deg:
                    candidates.append({
                        'ci': ci_info['ci'],
                        'node': node,
                        'missing_oct': ci_info['missing_oct']
                    })
                    break
        if len(candidates) >= n_sample:
            break
    
    print(f"  Sampling {len(candidates)} components")
    
    # Compute kernels
    kernels = {}
    for mc in candidates:
        _, i, j = mc['node']
        a = (basis[i] + basis[j]) / np.sqrt(2)
        L_a = np.zeros((level.dim, level.dim))
        for col in range(level.dim):
            L_a[:, col] = A.multiply(a, basis[col])
        _, S, Vt = np.linalg.svd(L_a)
        ker = Vt[S < 1e-8]
        kernels[mc['ci']] = {'kernel': ker, 'missing': mc['missing_oct'], 'dim': ker.shape[0]}
    
    # Pairwise overlaps
    print(f"\n  Pairwise kernel overlaps:")
    print(f"  {'Comp A':<15} {'Comp B':<15} {'Overlap':<10} {'Expected':<10}")
    print(f"  {'-'*50}")
    
    nonzero_count = 0
    total_nonmatched = 0
    cis = list(kernels.keys())
    
    for i_idx in range(len(cis)):
        for j_idx in range(i_idx+1, len(cis)):
            ci_a, ci_b = cis[i_idx], cis[j_idx]
            ka = kernels[ci_a]['kernel']
            kb = kernels[ci_b]['kernel']
            
            combined = np.vstack([ka, kb])
            rank = np.linalg.matrix_rank(combined, tol=1e-8)
            overlap = ka.shape[0] + kb.shape[0] - rank
            
            miss_a = kernels[ci_a]['missing']
            miss_b = kernels[ci_b]['missing']
            matched = miss_a == miss_b
            expected = "MATCH" if matched else "0"
            
            if not matched:
                total_nonmatched += 1
                if overlap > 0:
                    nonzero_count += 1
            
            if overlap > 0 or i_idx < 3:
                print(f"  C{ci_a}(m{miss_a}){'':<5} C{ci_b}(m{miss_b}){'':<5} "
                      f"{overlap:<10d} {expected}")
    
    print(f"\n  Non-matched pairs with nonzero overlap: {nonzero_count}/{total_nonmatched}")
    if nonzero_count == 0:
        print(f"  -> ZERO CROSSTALK: Addressing precision INTACT")
    else:
        print(f"  -> CROSSTALK DETECTED: Addressing precision DEGRADING")
    
    return nonzero_count, total_nonmatched


def test_transparency(level, n_sample=8):
    """Inter-component product norm test."""
    print(f"\n--- TRANSPARENCY TEST ({level.dim}D) ---")
    
    A = CayleyDickson(level.dim)
    basis = np.eye(level.dim)
    
    deviations = []
    for i_idx in range(min(n_sample, len(level.components))):
        for j_idx in range(i_idx+1, min(n_sample, len(level.components))):
            comp_a = level.components[i_idx]
            comp_b = level.components[j_idx]
            
            left_a = sorted([n for n in comp_a if n[0] == 'L'])[:2]
            left_b = sorted([n for n in comp_b if n[0] == 'L'])[:2]
            
            for na in left_a:
                for nb in left_b:
                    _, ia, ja = na
                    _, ib, jb = nb
                    a = basis[ia] + basis[ja]
                    b = basis[ib] + basis[jb]
                    a = a / np.sqrt(np.sum(a**2))
                    b = b / np.sqrt(np.sum(b**2))
                    ab = A.multiply(a, b)
                    norm = np.sum(ab**2)
                    deviations.append(abs(norm - 1.0))
    
    deviations = np.array(deviations)
    print(f"  Tested {len(deviations)} inter-component products")
    print(f"  Max deviation from 1.0: {deviations.max():.2e}")
    print(f"  Mean deviation: {deviations.mean():.2e}")
    
    if deviations.max() < 1e-8:
        print(f"  -> PERFECTLY TRANSPARENT")
    else:
        print(f"  -> TRANSPARENCY BROKEN")
    
    return deviations


def test_inheritance(level, prev_dim=None):
    """Verify previous level's ZD pairs are preserved."""
    if prev_dim is None:
        prev_dim = level.dim // 2
    
    print(f"\n--- INHERITANCE TEST ({prev_dim}D -> {level.dim}D) ---")
    
    within_low = [(i,j,k,l) for (i,j,k,l) in level.zd_pairs
                  if i < prev_dim and j < prev_dim and k < prev_dim and l < prev_dim]
    
    # Load previous level's count if available
    prev_dir = os.path.join(BASE_DIR, f"level_{prev_dim}D")
    expected = None
    if os.path.exists(os.path.join(prev_dir, 'metadata.json')):
        import json
        with open(os.path.join(prev_dir, 'metadata.json')) as f:
            meta = json.load(f)
        expected = meta['n_zd_pairs']
    
    print(f"  ZD pairs within {prev_dim}D subspace: {len(within_low)}")
    if expected is not None:
        print(f"  Expected (from {prev_dim}D level): {expected}")
        match = len(within_low) == expected
        print(f"  Match: {'YES' if match else 'NO'}")
    
    return len(within_low), expected


def test_kernels(level, n_per_stratum=2):
    """Sample kernel dimensions per stratum."""
    print(f"\n--- KERNEL SAMPLING ({level.dim}D) ---")
    
    A = CayleyDickson(level.dim)
    basis = np.eye(level.dim)
    
    for deg in sorted(level.degree_dist.keys()):
        samples = []
        for node in sorted(level.undirected_adj.keys()):
            if node[0] != 'L':
                continue
            if len(level.undirected_adj[node]) != deg:
                continue
            if len(samples) >= n_per_stratum:
                break
            samples.append(node)
        
        if not samples:
            continue
        
        print(f"\n  Degree {deg}:")
        for node in samples:
            _, i, j = node
            a = (basis[i] + basis[j]) / np.sqrt(2)
            L_a = np.zeros((level.dim, level.dim))
            for col in range(level.dim):
                L_a[:, col] = A.multiply(a, basis[col])
            sv = np.linalg.svd(L_a, compute_uv=False)
            kernel_dim = np.sum(sv < 1e-8)
            n_amplified = np.sum(sv > 1.2)
            n_normal = level.dim - kernel_dim - n_amplified
            print(f"    ({i:>2d},{j:>2d}): ker={kernel_dim}, "
                  f"normal={n_normal}, amplified={n_amplified}")


def test_hub_spoke(level):
    """Analyze mixed-degree component topology."""
    print(f"\n--- HUB-SPOKE ANALYSIS ({level.dim}D) ---")
    
    for ci_info in level.comp_info:
        degs = ci_info['deg_counts']
        if len(degs) < 2:
            continue
        
        min_deg = min(degs.keys())
        max_deg = max(degs.keys())
        
        if degs.get(min_deg, 0) > 5 and degs.get(max_deg, 0) <= 3:
            # Hub-spoke candidate
            comp = level.components[ci_info['ci']]
            left_nodes = sorted([n for n in comp if n[0] == 'L'])
            
            hubs = [n for n in left_nodes if len(level.undirected_adj[n]) == max_deg]
            spokes = [n for n in left_nodes if len(level.undirected_adj[n]) == min_deg]
            
            if not hubs:
                continue
            
            hub = hubs[0]
            hub_partners = set(level.undirected_adj[hub])
            
            spoke_in_hub = sum(1 for s in spokes
                              if set(level.undirected_adj[s]).issubset(hub_partners))
            
            print(f"  C{ci_info['ci']}: {len(spokes)}x d{min_deg} + {len(hubs)}x d{max_deg}")
            print(f"    Spokes within hub: {spoke_in_hub}/{len(spokes)}")
            
            if ci_info['ci'] > 5:  # limit output
                print(f"    ... (similar pattern continues)")
                break


def compare_levels(dim_a, dim_b):
    """Compare two levels side by side."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: {dim_a}D vs {dim_b}D")
    print(f"{'='*60}")
    
    la = load_level(dim_a)
    lb = load_level(dim_b)
    
    print(f"\n  {'Metric':<25} {dim_a:>10}D {dim_b:>10}D {'Ratio':>10}")
    print(f"  {'-'*57}")
    
    metrics = [
        ('ZD pairs', len(la.zd_pairs), len(lb.zd_pairs)),
        ('Components', len(la.components), len(lb.components)),
        ('Strata', len(la.degree_dist), len(lb.degree_dist)),
        ('Max degree', max(la.degree_dist.keys()), max(lb.degree_dist.keys())),
        ('Component types', len(set(c['sig'] for c in la.comp_info)),
                           len(set(c['sig'] for c in lb.comp_info))),
    ]
    
    for name, va, vb in metrics:
        ratio = vb / va if va > 0 else float('inf')
        print(f"  {name:<25} {va:>10} {vb:>10} {ratio:>10.2f}")


def summary_table():
    """Print summary table of all computed levels."""
    print(f"\n{'='*80}")
    print(f"CD TOWER SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n  {'Dim':>6} {'ZD pairs':>10} {'Growth':>8} {'Comps':>6} "
          f"{'Strata':>7} {'Types':>6} {'Max deg':>8}")
    print(f"  {'-'*55}")
    
    prev_zd = None
    dim = 16
    while True:
        save_dir = os.path.join(BASE_DIR, f"level_{dim}D")
        if not os.path.exists(os.path.join(save_dir, 'metadata.json')):
            break
        
        import json
        with open(os.path.join(save_dir, 'metadata.json')) as f:
            meta = json.load(f)
        
        n_zd = meta['n_zd_pairs']
        n_comp = meta['n_components']
        n_strata = meta['n_strata']
        max_deg = max(int(k) for k in meta['degree_dist'].keys())
        
        growth = f"x{n_zd/prev_zd:.1f}" if prev_zd else "-"
        
        # Count types from components.json
        n_types = "?"
        comp_file = os.path.join(save_dir, 'components.json')
        if os.path.exists(comp_file):
            with open(comp_file) as f:
                comps = json.load(f)
            sigs = set()
            for c in comps:
                sigs.add((c['sig'][0], tuple(tuple(x) for x in c['sig'][1])))
            n_types = len(sigs)
        
        print(f"  {dim:>6} {n_zd:>10} {growth:>8} {n_comp:>6} "
              f"{n_strata:>7} {n_types:>6} {max_deg:>8}")
        
        prev_zd = n_zd
        dim *= 2
    
    print(f"\n  Strata growth: ", end="")
    dim = 16
    strata_seq = []
    while True:
        save_dir = os.path.join(BASE_DIR, f"level_{dim}D")
        if not os.path.exists(os.path.join(save_dir, 'metadata.json')):
            break
        import json
        with open(os.path.join(save_dir, 'metadata.json')) as f:
            meta = json.load(f)
        strata_seq.append(str(meta['n_strata']))
        dim *= 2
    print(" -> ".join(strata_seq))


# ============================================================================
# MAIN
# ============================================================================

ALL_TESTS = {
    'summary': test_summary,
    'chirality': test_chirality,
    'overlap': test_overlap,
    'transparency': test_transparency,
    'inheritance': test_inheritance,
    'kernels': test_kernels,
    'hub_spoke': test_hub_spoke,
}

def main():
    args = sys.argv[1:]
    
    if not args:
        print("Usage:")
        print("  python analyze_level.py <dim>                   Run all tests")
        print("  python analyze_level.py <dim> --tests t1 t2     Run specific tests")
        print("  python analyze_level.py --compare <d1> <d2>     Compare levels")
        print("  python analyze_level.py --summary               Summary table")
        print(f"\nAvailable tests: {', '.join(ALL_TESTS.keys())}")
        return
    
    if args[0] == '--summary':
        summary_table()
        return
    
    if args[0] == '--compare':
        if len(args) >= 3:
            compare_levels(int(args[1]), int(args[2]))
        else:
            print("Usage: python analyze_level.py --compare <dim1> <dim2>")
        return
    
    dim = int(args[0])
    
    # Determine which tests to run
    if '--tests' in args:
        idx = args.index('--tests')
        test_names = args[idx+1:]
    else:
        test_names = list(ALL_TESTS.keys())
    
    # Load level
    level = load_level(dim)
    
    # Run tests
    for name in test_names:
        if name in ALL_TESTS:
            try:
                ALL_TESTS[name](level)
            except Exception as e:
                print(f"\n  TEST {name} FAILED: {e}")
        else:
            print(f"  Unknown test: {name}")


if __name__ == '__main__':
    main()
