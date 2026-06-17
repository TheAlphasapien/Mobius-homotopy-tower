#!/usr/bin/env python3
"""
heritage_edges.py — Cross-heritage edge structure at magic number transitions
===============================================================================

At each A in the greedy build, tracks:
1. The newly added element's heritage layer
2. Its connections to each heritage layer (within vs cross)
3. The cumulative cross-heritage edge matrix (layer × layer)
4. Transition signatures at magic numbers

The remainders (0, 0, 4, 12, 18, 18, 0) should be derivable from
the cross-heritage connection pattern at each transition.

USAGE:
  python heritage_edges.py --data-dir tower_data/level_1024D
"""

import os, sys, json, argparse, time, math
import numpy as np
from collections import defaultdict, Counter

MAGIC = {2, 8, 20, 28, 50, 82, 126}
MAGIC_LIST = [2, 8, 20, 28, 50, 82, 126]


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


def build_adjacency(elements, target, sign_arr):
    n = len(elements)
    adj = defaultdict(set)
    for a in range(n):
        i1,j1 = elements[a]
        for b in range(a+1,n):
            i2,j2 = elements[b]
            t1,s1=int(target[i1,i2]),int(sign_arr[i1,i2])
            t2,s2=int(target[i1,j2]),int(sign_arr[i1,j2])
            t3,s3=int(target[j1,i2]),int(sign_arr[j1,i2])
            t4,s4=int(target[j1,j2]),int(sign_arr[j1,j2])
            if (t1==t4 and s1==-s4 and t2==t3 and s2==-s3) or \
               (t1==t3 and s1==-s3 and t2==t4 and s2==-s4) or \
               (t1==t2 and s1==-s2 and t3==t4 and s3==-s4):
                adj[a].add(b);adj[b].add(a)
    return adj


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
    print(f"CROSS-HERITAGE EDGE ANALYSIS")
    print(f"Data: {args.data_dir}")
    print("="*70)

    target, sign_arr = load_arrays(args.data_dir)
    dim = target.shape[0]; D = dim // 2
    elements = load_component(args.data_dir)
    n = len(elements)

    # Heritage
    heritage = {}
    for idx, (i, j) in enumerate(elements):
        heritage[idx] = max(cd_layer(i), cd_layer(j - D))

    h_counts = Counter(heritage.values())
    h_layers = sorted(h_counts.keys())
    print(f"\n  {dim}D, {n} elements, heritage layers: {h_layers}")
    for h in h_layers:
        print(f"    H{h}: {h_counts[h]} elements")

    # Adjacency
    print(f"  Building adjacency...", flush=True)
    t0 = time.time()
    adj = build_adjacency(elements, target, sign_arr)
    print(f"  Done in {time.time()-t0:.1f}s")

    order = greedy_ordering(n, adj)

    # Track through build
    print(f"\n{'='*70}")
    print(f"ELEMENT-BY-ELEMENT HERITAGE EDGE ANALYSIS")
    print(f"{'='*70}")

    current_set = set()
    cumul_heritage = Counter()

    # Cross-heritage edge matrix (cumulative)
    n_layers = len(h_layers)
    h_idx = {h: i for i, h in enumerate(h_layers)}
    edge_matrix = np.zeros((n_layers, n_layers), dtype=int)

    print(f"\n  {'A':>4} {'H_new':>5} {'Conn':>5} {'Within':>7} {'Cross':>7} "
          f"{'CrossFrac':>10} {'Note':>12}")
    print(f"  {'─'*55}")

    # Detailed tracking
    results = []

    for step, idx in enumerate(order):
        A = step + 1
        h_new = heritage[idx]
        cumul_heritage[h_new] += 1
        
        # Count connections to existing elements by heritage layer
        conn_by_layer = Counter()
        for nb in adj[idx]:
            if nb in current_set:
                conn_by_layer[heritage[nb]] += 1

        total_conn = sum(conn_by_layer.values())
        within = conn_by_layer.get(h_new, 0)
        cross = total_conn - within

        # Update edge matrix
        for h_nb, count in conn_by_layer.items():
            hi = h_idx[h_new]
            hj = h_idx[h_nb]
            edge_matrix[hi, hj] += count
            if hi != hj:
                edge_matrix[hj, hi] += count  # symmetric

        current_set.add(idx)

        cf = cross / total_conn if total_conn > 0 else 0
        note = "◆ MAGIC" if A in MAGIC else ""

        show = (A <= 5 or A in MAGIC or A-1 in MAGIC or A+1 in MAGIC
                or A == n)
        if show:
            print(f"  {A:>4} H{h_new:>3} {total_conn:>5} {within:>7} {cross:>7} "
                  f"{cf:>10.4f} {note:>12}")

        results.append({
            'A': A, 'heritage': h_new, 'conn': total_conn,
            'within': within, 'cross': cross,
            'conn_by_layer': dict(conn_by_layer),
        })

    # Cross-heritage edge matrix at magic numbers
    print(f"\n{'='*70}")
    print(f"CROSS-HERITAGE EDGE MATRIX AT MAGIC NUMBERS")
    print(f"{'='*70}")

    current_set2 = set()
    for step, idx in enumerate(order):
        A = step + 1
        current_set2.add(idx)

        if A in MAGIC:
            # Compute full edge matrix for first A elements
            mat = np.zeros((n_layers, n_layers), dtype=int)
            sub = list(current_set2)
            for a in sub:
                for b in adj[a]:
                    if b in current_set2 and a < b:
                        hi = h_idx[heritage[a]]
                        hj = h_idx[heritage[b]]
                        mat[hi, hj] += 1
                        mat[hj, hi] += 1

            print(f"\n  A = {A} (magic):")
            # Show as fractions of maximum possible
            print(f"  {'':>5}", end='')
            for h in h_layers:
                print(f"  H{h:>2}", end='')
            print()
            
            for i, hi in enumerate(h_layers):
                print(f"  H{hi:>2}:", end='')
                for j, hj in enumerate(h_layers):
                    if mat[i,j] > 0:
                        # Max possible edges between layer i and j
                        ni = cumul_heritage.get(hi, 0)  # wrong, need cumul at this A
                        print(f"  {mat[i,j]:>4}", end='')
                    else:
                        print(f"  {'·':>4}", end='')
                print()

    # Transition analysis: what changes at each magic number?
    print(f"\n{'='*70}")
    print(f"TRANSITION ANALYSIS AT MAGIC NUMBERS")
    print(f"{'='*70}")

    # Re-build heritage counts at each magic number
    cumul3 = Counter()
    current3 = set()
    
    for step, idx in enumerate(order):
        A = step + 1
        h = heritage[idx]
        cumul3[h] += 1
        current3.add(idx)
        
        if A in MAGIC:
            print(f"\n  A = {A}:")
            
            # Heritage recipe
            recipe = [(hh, cumul3.get(hh, 0), h_counts[hh]) for hh in h_layers]
            active = [(hh, c, t) for hh, c, t in recipe if c > 0]
            
            for hh, c, t in active:
                fill = c/t
                status = "SETTLED" if fill >= 0.24 else "entering"
                
                # Count cross-heritage edges from this layer to all others
                layer_elems = [e for e in current3 if heritage[e] == hh]
                cross_from_layer = 0
                within_layer = 0
                for e in layer_elems:
                    for nb in adj[e]:
                        if nb in current3:
                            if heritage[nb] == hh:
                                within_layer += 1
                            else:
                                cross_from_layer += 1
                within_layer //= 2  # counted twice
                
                print(f"    H{hh}: {c}/{t} ({fill:.0%}) {status:>8} — "
                      f"within={within_layer}, cross={cross_from_layer}")
            
            # Total edges
            total_edges = sum(len(adj[e] & current3) for e in current3) // 2
            possible = A * (A-1) // 2
            print(f"    Total: {total_edges}/{possible} edges "
                  f"({100*total_edges/possible:.0f}% complete)")

    # The key question: what determines the remainder?
    print(f"\n{'='*70}")
    print(f"REMAINDER ANALYSIS")
    print(f"{'='*70}")
    
    cumul4 = Counter()
    for step, idx in enumerate(order):
        A = step + 1
        cumul4[heritage[idx]] += 1
        
        if A in MAGIC:
            settled_sum = sum(c for h, c in cumul4.items() 
                           if c / h_counts[h] >= 0.24)
            remainder = A - settled_sum
            
            # What heritage layers contribute to the remainder?
            remainder_layers = [(h, c) for h, c in cumul4.items()
                              if c / h_counts[h] < 0.24 and c > 0]
            
            print(f"  A={A:>4}: settled={settled_sum}, remainder={remainder}")
            if remainder_layers:
                print(f"          remainder from: {remainder_layers}")

    # Save
    outpath = f"heritage_edges_{dim}D.csv"
    with open(outpath, 'w') as f:
        f.write("A,heritage,conn,within,cross,cross_frac,is_magic\n")
        for r in results:
            cf = r['cross']/r['conn'] if r['conn'] > 0 else 0
            m = 1 if r['A'] in MAGIC else 0
            f.write(f"{r['A']},{r['heritage']},{r['conn']},"
                    f"{r['within']},{r['cross']},{cf:.6f},{m}\n")
    print(f"\n  Saved: {outpath}")


if __name__ == "__main__":
    main()
