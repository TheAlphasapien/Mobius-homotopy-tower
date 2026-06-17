#!/usr/bin/env python3
"""
heritage_decomp.py — Heritage layer decomposition through greedy build
========================================================================

Tracks which CD heritage layers contribute elements to the nucleus
at each A. Checks whether magic numbers correspond to specific
heritage balance conditions.

Key question: do higher tower levels bring MORE heritage layers into
the pure-degree component, and does this change the magic number signal?

USAGE:
  python heritage_decomp.py --data-dir tower_lean/level_1024D
  python heritage_decomp.py --data-dir tower_lean/level_2048D
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
    print(f"HERITAGE DECOMPOSITION — {os.path.basename(args.data_dir)}")
    print("="*70)

    target, sign_arr = load_arrays(args.data_dir)
    dim = target.shape[0]; D = dim // 2
    level = int(math.log2(dim))
    elements = load_component(args.data_dir)
    n = len(elements)

    print(f"  {dim}D, D={D}, level={level}")
    print(f"  Pure component: {n} elements")

    # Heritage for each element
    heritage = {}
    for idx, (i, j) in enumerate(elements):
        jp = j - D
        heritage[idx] = max(cd_layer(i), cd_layer(jp))

    h_counts = Counter(heritage.values())
    h_layers = sorted(h_counts.keys())
    print(f"  Heritage layers present: {h_layers}")
    print(f"  Elements per layer:")
    for h in h_layers:
        print(f"    Layer {h}: {h_counts[h]} elements "
              f"(indices {2**h}..{2**(h+1)-1})")

    # Build adjacency
    print(f"  Building adjacency...", flush=True)
    t0 = time.time()
    adj = build_adjacency(elements, target, sign_arr)
    print(f"  {sum(len(v) for v in adj.values())//2} edges in {time.time()-t0:.1f}s")

    order = greedy_ordering(n, adj)

    # Track heritage through build
    print(f"\n{'='*70}")
    print(f"HERITAGE RECIPE THROUGH GREEDY BUILD")
    print(f"{'='*70}")

    # Header
    print(f"\n{'A':>4}", end='')
    for h in h_layers:
        print(f"  H{h:>2}", end='')
    print(f"  {'Balance':>8} {'CrossFrac':>10} {'Note':>12}")
    print("─"*70)

    cumul = Counter()
    
    for step in range(n):
        idx = order[step]
        A = step + 1
        h = heritage[idx]
        cumul[h] += 1

        # Heritage balance: std of fill fractions
        fill_fracs = [cumul.get(hh, 0) / h_counts[hh] for hh in h_layers]
        balance = np.std(fill_fracs)

        # Cross-heritage edges
        if A in MAGIC or A in {3,4,5,10,15,30,40,60,100,n} or A <= 5:
            sub = set(order[:A])
            within = cross = 0
            for e1 in sub:
                for e2 in adj[e1]:
                    if e2 in sub and e1 < e2:
                        if heritage[e1] == heritage[e2]: within += 1
                        else: cross += 1
            total_e = within + cross
            cf = cross / total_e if total_e > 0 else 0
        else:
            cf = -1  # not computed

        note = "◆ MAGIC" if A in MAGIC else ""

        show = (A <= 5 or A in MAGIC or A-1 in MAGIC or A+1 in MAGIC
                or A == n or A % 20 == 0)

        if show:
            print(f"{A:>4}", end='')
            for hh in h_layers:
                c = cumul.get(hh, 0)
                print(f"  {c:>4}", end='')
            print(f"  {balance:>8.4f}", end='')
            if cf >= 0:
                print(f"  {cf:>10.4f}", end='')
            else:
                print(f"  {'':>10}", end='')
            print(f"  {note:>12}")

    # Magic number heritage balance analysis
    print(f"\n{'='*70}")
    print(f"HERITAGE BALANCE AT MAGIC NUMBERS")
    print(f"{'='*70}")

    cumul2 = Counter()
    balance_at = {}
    exact_balance_A = []

    for step in range(n):
        idx = order[step]
        A = step + 1
        h = heritage[idx]
        cumul2[h] += 1

        counts = [cumul2.get(hh, 0) for hh in h_layers]
        # Check if all counts are equal
        if len(set(counts)) == 1 and A > 1:
            exact_balance_A.append(A)

        if A in MAGIC:
            balance_at[A] = dict(cumul2)

    print(f"\n  A values with EXACT heritage balance (all layers equal count):")
    print(f"  {exact_balance_A[:30]}{'...' if len(exact_balance_A) > 30 else ''}")
    magic_balanced = [a for a in exact_balance_A if a in MAGIC]
    print(f"  Of which are magic: {magic_balanced}")

    print(f"\n  Heritage counts at magic numbers:")
    for m in sorted(MAGIC):
        if m in balance_at:
            counts = {h: balance_at[m].get(h, 0) for h in h_layers}
            diffs = [abs(counts[h_layers[i]] - counts[h_layers[j]]) 
                    for i in range(len(h_layers)) for j in range(i+1, len(h_layers))]
            max_diff = max(diffs) if diffs else 0
            balanced = "EXACT" if max_diff == 0 else f"off by {max_diff}"
            print(f"    A={m:>4}: {dict(counts)} — {balanced}")

    # Save
    outpath = f"heritage_decomp_{dim}D.csv"
    cumul3 = Counter()
    with open(outpath, 'w') as f:
        headers = ['A', 'is_magic'] + [f'H{h}' for h in h_layers] + ['balance']
        f.write(','.join(headers) + '\n')
        for step in range(n):
            idx = order[step]
            A = step + 1
            h = heritage[idx]
            cumul3[h] += 1
            fill_fracs = [cumul3.get(hh, 0) / h_counts[hh] for hh in h_layers]
            bal = np.std(fill_fracs)
            m = 1 if A in MAGIC else 0
            vals = [str(A), str(m)] + [str(cumul3.get(hh,0)) for hh in h_layers] + [f'{bal:.6f}']
            f.write(','.join(vals) + '\n')
    print(f"\n  Saved: {outpath}")


if __name__ == "__main__":
    main()
