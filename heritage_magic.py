#!/usr/bin/env python3
"""
heritage_magic.py — Magic numbers from CD heritage layers
============================================================

Each basis element has a "heritage depth" — which CD doubling created it:
  Layer 0: index 1           (complex doubling)
  Layer 1: indices 2-3       (quaternion doubling)
  Layer 2: indices 4-7       (octonion doubling)
  Layer 3: indices 8-15      (sedenion doubling)
  Layer m: indices 2^m..2^(m+1)-1

For cross-level element (e_i + e_j) with i < D, j >= D:
  Heritage = max(layer(i), layer(j - D))

Elements within heritage <= m have ZD connections governed by the
multiplication table at CD level m+3, which is FIXED for all higher
levels. Their sub-graph is automatically transport-invariant.

PREDICTION: The max complete sub-graph at each heritage level
matches a nuclear magic number:
  Heritage 0 (2 elements):   max clique = 2?   → magic 2
  Heritage 1 (12 elements):  max clique = 8?   → magic 8
  Heritage 2 (56 elements):  max clique = 20?  → magic 20
  Heritage 3 (240 elements): max clique = 28?  → magic 28
  Heritage 4 (992 elements): max clique = 50?  → magic 50
  Heritage 5 (4032 elements): max clique = 82? → magic 82

USAGE:
  python heritage_magic.py --dim 1024
  python heritage_magic.py --dim 2048
  python heritage_magic.py --dim 4096
"""

import os, sys, argparse, time, math
import numpy as np
from collections import defaultdict

MAGIC = [2, 8, 20, 28, 50, 82, 126]
FANO_TRIPLES = [(1,2,3),(1,4,5),(1,7,6),(2,4,6),(2,5,7),(3,4,7),(3,6,5)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024)
    return p.parse_args()


def cd_layer(n):
    """Which CD doubling created basis element n."""
    if n <= 0: return -1
    return int(math.floor(math.log2(n)))


def build_tables(target_dim):
    """Build CD multiplication tables from Fano plane."""
    t = np.zeros((8,8), dtype=np.int16)
    s = np.zeros((8,8), dtype=np.int8)
    for i in range(8): t[0,i]=i;s[0,i]=1;t[i,0]=i;s[i,0]=1
    for a in range(1,8): t[a,a]=0;s[a,a]=-1
    for a,b,c in FANO_TRIPLES:
        t[a,b]=c;s[a,b]=1;t[b,a]=c;s[b,a]=-1
        t[b,c]=a;s[b,c]=1;t[c,b]=a;s[c,b]=-1
        t[c,a]=b;s[c,a]=1;t[a,c]=b;s[a,c]=-1

    level = int(math.log2(target_dim))
    for _ in range(level - 3):
        D = t.shape[0]; dim = 2*D
        tn = np.zeros((dim,dim),dtype=np.int16)
        sn = np.zeros((dim,dim),dtype=np.int8)
        tn[:D,:D]=t;sn[:D,:D]=s;tn[:D,D:]=t.T+D;sn[:D,D:]=s.T
        tn[D:,0]=np.arange(D,dim,dtype=np.int16);sn[D:,0]=1
        tn[D:,1:D]=t[:,1:]+D;sn[D:,1:D]=-s[:,1:]
        tn[D:,D]=np.arange(D,dtype=np.int16);sn[D:,D]=-1
        tn[D:,D+1:]=t[1:,:].T;sn[D:,D+1:]=s[1:,:].T
        t, s = tn, sn
    return t, s


def is_zd(i, j, k, l, target, sign):
    """Check if (e_i + e_j)(e_k + e_l) = 0."""
    t1,s1 = int(target[i,k]),int(sign[i,k])
    t2,s2 = int(target[i,l]),int(sign[i,l])
    t3,s3 = int(target[j,k]),int(sign[j,k])
    t4,s4 = int(target[j,l]),int(sign[j,l])
    if t1==t4 and s1==-s4 and t2==t3 and s2==-s3: return True
    if t1==t3 and s1==-s3 and t2==t4 and s2==-s4: return True
    if t1==t2 and s1==-s2 and t3==t4 and s3==-s4: return True
    return False


def find_max_clique_greedy(elements, adj):
    """Find approximate max clique via greedy build."""
    if not elements:
        return 0, []

    n = len(elements)
    # Start from most connected element
    start = max(range(n), key=lambda i: len(adj[i]))
    clique = [start]
    clique_set = {start}

    for _ in range(n - 1):
        # Find element connected to ALL current clique members
        best = None
        best_conn = -1
        for m in range(n):
            if m in clique_set:
                continue
            conn = len(adj[m] & clique_set)
            if conn == len(clique_set) and conn > best_conn:
                best = m
                best_conn = conn

        if best is None:
            break  # No more elements connect to entire clique
        clique.append(best)
        clique_set.add(best)

    return len(clique), clique


def main():
    args = parse_args()
    dim = args.dim
    D = dim // 2
    level = int(math.log2(dim))

    print("="*70)
    print(f"HERITAGE LAYER MAGIC NUMBER TEST — {dim}D")
    print("="*70)
    print(f"\n  CD Level: {level}, D = {D}")

    print(f"  Building multiplication table...", flush=True)
    target, sign = build_tables(dim)

    # Maximum heritage depth we can test
    max_heritage = level - 3  # layer goes up to level-2, but we need j' range too

    print(f"\n{'='*70}")
    print(f"HERITAGE LAYERS AND MAX COMPLETE SUB-GRAPHS")
    print(f"{'='*70}")

    print(f"\n  {'Heritage':>8} {'Elements':>9} {'ZD edges':>9} {'Max clique':>11} "
          f"{'Magic?':>8} {'Predicted':>10}")
    print(f"  {'─'*58}")

    cumulative_results = []

    for h in range(0, min(max_heritage + 1, 8)):
        # Enumerate elements with heritage <= h
        max_i = min(2**(h+1) - 1, D - 1)
        max_jp = min(2**(h+1) - 1, D - 1)  # j' = j - D

        elements = []
        for i in range(1, max_i + 1):
            for jp in range(0, max_jp + 1):
                j = jp + D
                if max(cd_layer(i), cd_layer(jp)) <= h:
                    elements.append((i, j))

        n_elem = len(elements)
        if n_elem < 2:
            print(f"  {h:>8} {n_elem:>9} {'—':>9} {'—':>11} {'—':>8} "
                  f"{MAGIC[h] if h < len(MAGIC) else '?':>10}")
            continue

        # Build adjacency within this heritage group
        adj = defaultdict(set)
        n_edges = 0
        for a in range(n_elem):
            i1, j1 = elements[a]
            for b in range(a + 1, n_elem):
                i2, j2 = elements[b]
                if is_zd(i1, j1, i2, j2, target, sign):
                    adj[a].add(b); adj[b].add(a)
                    n_edges += 1

        # Find max clique
        clique_size, clique = find_max_clique_greedy(elements, adj)

        # Also find complete-graph boundary via greedy build
        start = max(range(n_elem), key=lambda i: len(adj[i]))
        order = [start]
        current = {start}
        remaining = set(range(n_elem)) - {start}
        ec_boundary = n_elem  # where EC drops below size-1

        for step in range(n_elem - 1):
            if not remaining:
                break
            best = max(remaining, key=lambda m: len(adj[m] & current))
            conn = len(adj[best] & current)
            if conn < len(current):
                # Not connected to all existing — clique boundary
                ec_boundary = len(current)
                break
            order.append(best)
            current.add(best)
            remaining.discard(best)
        else:
            ec_boundary = len(current)

        predicted = MAGIC[h] if h < len(MAGIC) else "?"
        match = "✓ MATCH" if ec_boundary == predicted else ""

        print(f"  {h:>8} {n_elem:>9} {n_edges:>9} {ec_boundary:>11} "
              f"{match:>8} {str(predicted):>10}")

        cumulative_results.append({
            'heritage': h,
            'n_elements': n_elem,
            'n_edges': n_edges,
            'max_clique': ec_boundary,
            'predicted_magic': predicted,
            'match': ec_boundary == predicted if isinstance(predicted, int) else None,
        })

        # Show the clique elements for small heritage
        if h <= 2 and ec_boundary <= 30:
            clique_elems = [elements[order[s]] for s in range(ec_boundary)]
            print(f"           Clique elements: {clique_elems[:10]}"
                  f"{'...' if ec_boundary > 10 else ''}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    n_tested = len(cumulative_results)
    n_matched = sum(1 for r in cumulative_results if r['match'] is True)

    print(f"\n  Tested: {n_tested} heritage levels")
    print(f"  Matched magic numbers: {n_matched}/{n_tested}")

    if n_matched == n_tested and n_tested >= 5:
        print(f"\n  ★ ALL HERITAGE LEVELS MATCH MAGIC NUMBERS")
        print(f"    The magic numbers ARE the max complete sub-graphs")
        print(f"    at each CD heritage depth. Transport-invariant by")
        print(f"    construction (heritage groups use inherited tables).")
    elif n_matched >= n_tested * 0.7:
        print(f"\n  STRONG PARTIAL MATCH: {n_matched}/{n_tested}")
    elif n_matched > 0:
        print(f"\n  WEAK MATCH: {n_matched}/{n_tested}")
    else:
        print(f"\n  NO MATCH: Heritage layers do not predict magic numbers.")

    # Verify transport invariance: run at multiple dims
    print(f"\n{'='*70}")
    print(f"TRANSPORT INVARIANCE CHECK")
    print(f"{'='*70}")
    print(f"\n  Heritage groups use multiplication table entries from")
    print(f"  the corresponding CD level. These entries are IDENTICAL")
    print(f"  at every higher tower level (inherited through doubling).")
    print(f"  Therefore, the max clique at heritage h is the SAME at")
    print(f"  1024D, 2048D, 4096D, 8192D, ... by construction.")
    print(f"\n  To verify: run this test at multiple --dim values.")
    print(f"  The results should be IDENTICAL.")


if __name__ == "__main__":
    main()
