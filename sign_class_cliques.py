#!/usr/bin/env python3
"""
sign_class_cliques.py — Entering clique structure from sign classes
=====================================================================

The 25% heritage fill is derived: 2 independent sign conditions from 
the 4-term cancellation give 4 equal classes. This script:

1. Classifies elements by heritage AND sign class
2. Follows the greedy build step by step
3. At each A, reports the sign-class composition of the entering layer
4. Identifies WHY the entering clique sizes are 4, 12, 18, 18, 62

The entering clique = elements from the new heritage layer that are
in the compatible sign class AND form a complete sub-graph with 
everything already in the build.

The sub-clique sizes (4, 12, 18, 18, 62) should correspond to 
structural sub-divisions within each compatible set.

USAGE:
  python sign_class_cliques.py --data-dir tower_data/level_1024D
"""

import os, sys, json, argparse, time, math
import numpy as np
from collections import defaultdict, Counter

MAGIC = {2, 8, 20, 28, 50, 82, 126}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    return p.parse_args()


def cd_layer(n):
    if n <= 0: return -1
    return int(math.floor(math.log2(n)))


def load_arrays(data_dir):
    for prefix in ['target', 'target_512', 'target_1024', 'target_2048',
                    'target_4096']:
        path = os.path.join(data_dir, f"{prefix}.npy")
        if os.path.exists(path):
            return np.load(path), np.load(path.replace('target','sign'))
    mult_path = os.path.join(data_dir, "mult_table.npy")
    if os.path.exists(mult_path):
        M = np.load(mult_path); D = M.shape[0]
        t = np.zeros((D,D),dtype=np.int16); s = np.zeros((D,D),dtype=np.int8)
        for i in range(D):
            for j in range(D):
                k = np.argmax(np.abs(M[i,j,:])); t[i,j]=k
                s[i,j] = 1 if M[i,j,k]>0 else -1
        return t, s
    sys.exit(f"No arrays found in {data_dir}")


def load_component(data_dir):
    with open(os.path.join(data_dir, "components.json")) as f:
        comps = json.load(f)
    pure = [c for c in comps if len(c['deg_counts'])==1]
    c = max(pure, key=lambda x: x['n_left']) if pure else max(comps, key=lambda x: x['n_left'])
    return [tuple(p) for p in c['left_pairs']]


def build_adjacency_with_signs(elements, target, sign_arr):
    """Build adjacency AND store the two sign values for each ZD pair."""
    n = len(elements)
    D = target.shape[0] // 2
    adj = defaultdict(set)
    pair_signs = {}  # (a_idx, b_idx) -> (s1, s2)
    
    for a in range(n):
        i1, j1 = elements[a]
        for b in range(a+1, n):
            i2, j2 = elements[b]
            t1 = int(target[i1,i2]); s1 = int(sign_arr[i1,i2])
            t2 = int(target[i1,j2]); s2 = int(sign_arr[i1,j2])
            t3 = int(target[j1,i2]); s3 = int(sign_arr[j1,i2])
            t4 = int(target[j1,j2]); s4 = int(sign_arr[j1,j2])
            
            if t1==t4 and s1==-s4 and t2==t3 and s2==-s3:
                adj[a].add(b); adj[b].add(a)
                pair_signs[(a,b)] = (s1, s2)
            elif t1==t3 and s1==-s3 and t2==t4 and s2==-s4:
                adj[a].add(b); adj[b].add(a)
                pair_signs[(a,b)] = (s1, s2)
            elif t1==t2 and s1==-s2 and t3==t4 and s3==-s4:
                adj[a].add(b); adj[b].add(a)
                pair_signs[(a,b)] = (s1, s3)
    
    return adj, pair_signs


def greedy_ordering(n, adj):
    start = max(range(n), key=lambda i: len(adj[i]))
    order = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        cs = set(order)
        best = max(remaining, key=lambda m: sum(1 for nb in adj[m] if nb in cs))
        order.append(best)
        remaining.discard(best)
    return order


def main():
    args = parse_args()

    print("="*70)
    print("SIGN CLASS ENTERING CLIQUE ANALYSIS")
    print(f"Data: {args.data_dir}")
    print("="*70)

    target, sign_arr = load_arrays(args.data_dir)
    dim = target.shape[0]; D = dim // 2
    elements = load_component(args.data_dir)
    n = len(elements)

    heritage = {}
    for idx, (i, j) in enumerate(elements):
        heritage[idx] = max(cd_layer(i), cd_layer(j - D))

    h_counts = Counter(heritage.values())
    h_layers = sorted(h_counts.keys())
    print(f"\n  {dim}D, {n} elements, layers: {dict(h_counts)}")

    # Build adjacency with sign pairs
    print(f"  Building adjacency with signs...", flush=True)
    t0 = time.time()
    adj, pair_signs = build_adjacency_with_signs(elements, target, sign_arr)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Verify sign class distribution within each layer
    print(f"\n{'='*70}")
    print(f"SIGN CLASS DISTRIBUTION (within-layer)")
    print(f"{'='*70}")

    layer_elems = defaultdict(list)
    for idx in range(n):
        layer_elems[heritage[idx]].append(idx)

    for h in h_layers:
        elems = layer_elems[h]
        signs = []
        for a in range(len(elems)):
            for b in range(a+1, len(elems)):
                key = (min(elems[a], elems[b]), max(elems[a], elems[b]))
                if key in pair_signs:
                    signs.append(pair_signs[key])
        
        if signs:
            sc = Counter(signs)
            total = len(signs)
            print(f"\n  H{h} ({len(elems)} elements, {total} pairs):")
            for sig in sorted(sc.keys()):
                print(f"    {sig}: {sc[sig]:>5} ({100*sc[sig]/total:.1f}%)")

    # Greedy build with detailed entering layer tracking
    print(f"\n{'='*70}")
    print(f"GREEDY BUILD — ENTERING LAYER DETAIL")
    print(f"{'='*70}")

    order = greedy_ordering(n, adj)
    current_set = set()
    cumul = Counter()

    # Track: for the entering layer, which elements are in and 
    # what's their internal clique structure?
    current_entering_layer = None
    entering_elements = []

    print(f"\n  {'A':>4} {'H':>3} {'Conn':>5} {'InLayer':>8} {'EnterSize':>10} "
          f"{'EnterCliq':>10} {'Note':>12}")
    print(f"  {'─'*60}")

    for step, idx in enumerate(order):
        A = step + 1
        h = heritage[idx]
        cumul[h] += 1
        current_set.add(idx)

        # Detect when we enter a new heritage layer
        if h != current_entering_layer:
            if current_entering_layer is not None and A > 2:
                # Layer transition
                pass
            current_entering_layer = h
            entering_elements = [idx]
        else:
            entering_elements.append(idx)

        # Count edges within entering elements
        enter_set = set(entering_elements)
        within_enter = sum(1 for a in enter_set for b in adj[a] 
                         if b in enter_set and a < b)
        max_cliq = len(entering_elements)
        possible = max_cliq * (max_cliq - 1) // 2
        is_clique = within_enter == possible and possible > 0

        note = ""
        if A in MAGIC: note = "◆ MAGIC"

        conn = len(adj[idx] & (current_set - {idx}))

        show = (A <= 10 or A in MAGIC or A-1 in MAGIC or A+1 in MAGIC
                or not is_clique or A == n)

        if show:
            cliq_status = "CLIQUE" if is_clique else f"{within_enter}/{possible}"
            print(f"  {A:>4} H{h:>2} {conn:>5} "
                  f"{cumul[h]:>4}/{h_counts[h]:<4}"
                  f"{len(entering_elements):>10} "
                  f"{cliq_status:>10} {note:>12}")

    # Detailed analysis at magic numbers
    print(f"\n{'='*70}")
    print(f"ENTERING LAYER STATE AT EACH MAGIC NUMBER")
    print(f"{'='*70}")

    current_set2 = set()
    cumul2 = Counter()

    for step, idx in enumerate(order):
        A = step + 1
        h = heritage[idx]
        cumul2[h] += 1
        current_set2.add(idx)

        if A in MAGIC:
            # Find which layers are "entering" (< 25% fill)
            entering_layers = []
            settled_layers = []
            for hh in h_layers:
                quarter = max(1, h_counts[hh] // 4)
                if cumul2.get(hh, 0) >= quarter:
                    settled_layers.append(hh)
                elif cumul2.get(hh, 0) > 0:
                    entering_layers.append(hh)

            print(f"\n  A = {A} (magic):")
            print(f"    Settled: {settled_layers} (sum = {sum(cumul2[h] for h in settled_layers)})")
            
            for eh in entering_layers:
                e_elems = [e for e in current_set2 if heritage[e] == eh]
                n_enter = len(e_elems)
                
                # Check if entering elements form a clique
                within = sum(1 for a in e_elems for b in adj[a] 
                           if b in set(e_elems) and a < b)
                possible = n_enter * (n_enter - 1) // 2
                
                # Sign class analysis within entering elements
                enter_signs = []
                for a in range(len(e_elems)):
                    for b in range(a+1, len(e_elems)):
                        key = (min(e_elems[a], e_elems[b]), 
                               max(e_elems[a], e_elems[b]))
                        if key in pair_signs:
                            enter_signs.append(pair_signs[key])
                
                sign_dist = Counter(enter_signs)
                
                print(f"    Entering H{eh}: {n_enter}/{h_counts[eh]} "
                      f"({100*n_enter/h_counts[eh]:.0f}%)")
                print(f"      Within edges: {within}/{possible} "
                      f"({'CLIQUE' if within == possible else 'NOT CLIQUE'})")
                if sign_dist:
                    print(f"      Sign pairs: {dict(sign_dist)}")

    # Save
    outpath = f"sign_class_cliques_{dim}D.csv"
    cumul3 = Counter()
    with open(outpath, 'w') as f:
        f.write("A,heritage,conn,layer_count,layer_total,is_magic\n")
        for step, idx in enumerate(order):
            A = step + 1
            h = heritage[idx]
            cumul3[h] += 1
            conn = len(adj[idx] & set(order[:step]))
            m = 1 if A in MAGIC else 0
            f.write(f"{A},{h},{conn},{cumul3[h]},{h_counts[h]},{m}\n")
    print(f"\n  Saved: {outpath}")


if __name__ == "__main__":
    main()
