#!/usr/bin/env python3
"""
viz_export.py — Export GraphML/CSV artifacts for a computed level.

Usage examples:
  # Full-export: bipartite + component meta-graph
  python viz_export.py tower_data/level_256D --out figs/256D --graphml bipartite components

  # Component meta-graph only, connecting components that share the same signature
  python viz_export.py tower_data/level_256D --out figs/256D --graphml components --connect-by signature

  # Connect-by using degree profile instead of 'sig'
  python viz_export.py tower_data/level_256D --out figs/256D --graphml components --connect-by deg-profile

  # Focused bipartite subgraph (only these components), and sample 200k edges for speed
  python viz_export.py tower_data/level_512D --out figs/512D --graphml bipartite \
    --subgraph C12 C47 --sample 200000 --embed-component-ids
"""

import os
import json
import argparse
import random
import numpy as np


# ---------------------------
#           I/O
# ---------------------------

def ensure_out(d):
    os.makedirs(d, exist_ok=True)

def load_level(level_dir):
    meta = json.load(open(os.path.join(level_dir, 'metadata.json'), 'r'))
    comps = json.load(open(os.path.join(level_dir, 'components.json'), 'r'))
    edges = np.load(os.path.join(level_dir, 'edges.npy'))  # (E, 4): Li, Lj, Rk, Rl
    return meta, comps, edges


# ---------------------------
#     Helper: hashable sig
# ---------------------------

def hashable_signature(comp):
    """
    Build a hashable key that represents the component's degree-profile signature.
    Works whether 'sig' is present as JSON lists or we only have a deg_counts dict.
    Returns a tuple of (deg, count) int pairs, sorted; or () if nothing is available.
    """
    # Try 'sig' first: expected like [size, [[deg, count], [deg, count], ...]]
    raw = None
    if 'sig' in comp and isinstance(comp['sig'], (list, tuple)) and len(comp['sig']) >= 2:
        raw = comp['sig'][1]
    elif 'deg_counts' in comp and isinstance(comp['deg_counts'], dict):
        # fallback from dict
        raw = sorted((int(k), int(v)) for k, v in comp['deg_counts'].items())

    if raw is None:
        return ()

    items = []
    if isinstance(raw, dict):
        items = sorted((int(k), int(v)) for k, v in raw.items())
    else:
        # Expect a list/tuple of pairs (maybe nested lists); normalize and sort.
        for pair in raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                a, b = pair
                try:
                    items.append((int(a), int(b)))
                except Exception:
                    # Ignore malformed pairs
                    continue
        items.sort()
    return tuple(items)


# ---------------------------
#   Components Graph Export
# ---------------------------

def export_components_graphml(level_dir, out_dir, connect_by=None):
    """
    Export the components meta-graph as GraphML (or CSV fallback).
    connect_by: None | 'signature' | 'deg-profile'
    """
    ensure_out(out_dir)
    try:
        import networkx as nx
    except Exception:
        return export_components_csv(level_dir, out_dir)

    meta, comps, edges = load_level(level_dir)

    G = nx.Graph()
    for c in comps:
        # Robust deg_mean
        deg_counts = c.get('deg_counts', {})
        if not isinstance(deg_counts, dict):
            deg_counts = {}
        total = sum(int(v) for v in deg_counts.values())
        deg_mean = 0.0
        if total > 0:
            deg_mean = sum(int(k) * int(v) for k, v in deg_counts.items()) / total

        # Safe accessors
        ci = int(c.get('ci', -1))
        size = int(c.get('size', 0))
        n_left = int(c.get('n_left', 0))
        n_right = int(c.get('n_right', 0))
        missing_oct = c.get('missing_oct', []) or []
        n_low = int(c.get('n_low', 0))
        n_high = int(c.get('n_high', 0))
        sig_size = 0
        sig = c.get('sig')
        if isinstance(sig, (list, tuple)) and len(sig) >= 1:
            try:
                sig_size = int(sig[0])
            except Exception:
                sig_size = 0

        G.add_node(
            f"C{ci}",
            size=size, n_left=n_left, n_right=n_right,
            deg_mean=float(deg_mean), missing_oct_count=int(len(missing_oct)),
            n_low=n_low, n_high=n_high,
            sig_size=sig_size
        )

    # Optional: connect components that share the same "bin"
    if connect_by in ('signature', 'deg-profile'):
        bins = {}
        for c in comps:
            ci = int(c.get('ci', -1))
            if connect_by == 'deg-profile':
                # derive from deg_counts only
                dc = c.get('deg_counts', {})
                if not isinstance(dc, dict):
                    key = ()
                else:
                    key = tuple(sorted((int(k), int(v)) for k, v in dc.items()))
            else:
                # connect_by == 'signature'
                key = hashable_signature(c)
            bins.setdefault(key, []).append(ci)

        # Add undirected clique per bin
        for key, lst in bins.items():
            if len(lst) > 1:
                for i in range(len(lst) - 1):
                    for j in range(i + 1, len(lst)):
                        G.add_edge(f"C{lst[i]}", f"C{lst[j]}")

    path = os.path.join(out_dir, 'components.graphml')
    try:
        nx.write_graphml(G, path)
        return path
    except Exception:
        return export_components_csv(level_dir, out_dir)


def export_components_csv(level_dir, out_dir):
    ensure_out(out_dir)
    meta, comps, edges = load_level(level_dir)
    outp = os.path.join(out_dir, 'components_nodes.csv')
    with open(outp, 'w', encoding='utf-8') as f:
        f.write('id,size,n_left,n_right,deg_mean,missing_oct_count,n_low,n_high\n')
        for c in comps:
            deg_counts = c.get('deg_counts', {})
            if not isinstance(deg_counts, dict):
                deg_counts = {}
            total = sum(int(v) for v in deg_counts.values())
            deg_mean = (sum(int(k) * int(v) for k, v in deg_counts.items()) / total) if total > 0 else 0
            f.write(
                f"C{int(c.get('ci', -1))},"
                f"{int(c.get('size', 0))},"
                f"{int(c.get('n_left', 0))},"
                f"{int(c.get('n_right', 0))},"
                f"{deg_mean},"
                f"{len(c.get('missing_oct', []) or [])},"
                f"{int(c.get('n_low', 0))},"
                f"{int(c.get('n_high', 0))}\n"
            )
    return outp


# ---------------------------
#      Bipartite Export
# ---------------------------

def build_pair_to_component_maps(comps):
    """
    Build dicts that map:
      ('L', i, j) -> ci
      ('R', k, l) -> ci
    using left_pairs/right_pairs provided in components.json (no recompute).
    """
    lmap = {}
    rmap = {}
    for c in comps:
        ci = int(c.get('ci', -1))
        for p in c.get('left_pairs', []) or []:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                i, j = int(p[0]), int(p[1])
                lmap[('L', i, j)] = ci
        for p in c.get('right_pairs', []) or []:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                k, l = int(p[0]), int(p[1])
                rmap[('R', k, l)] = ci
    return lmap, rmap


def filter_edges_by_components(edges, lmap, rmap, keep_cis):
    """
    Keep only those edges (Li, Lj, Rk, Rl) where BOTH the left pair's ci and the right pair's ci
    are in keep_cis (i.e., edges that belong to the selected components).
    """
    keep = set(int(x) for x in keep_cis)
    mask = []
    for (Li, Lj, Rk, Rl) in edges:
        ciL = lmap.get(('L', int(Li), int(Lj)), None)
        ciR = rmap.get(('R', int(Rk), int(Rl)), None)
        if ciL is None or ciR is None:
            # If either side has no mapping, conservatively drop (shouldn't happen for computed levels)
            continue
        if (ciL in keep) and (ciR in keep):
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask, dtype=bool)
    return edges[mask]


def sample_edges(edges, sample_n):
    """
    Randomly sample up to sample_n edges uniformly.
    """
    E = edges.shape[0]
    if sample_n is None or sample_n <= 0 or sample_n >= E:
        return edges
    idx = np.arange(E)
    np.random.shuffle(idx)
    idx = idx[:sample_n]
    idx.sort()
    return edges[idx]


def export_bipartite_graphml(level_dir, out_dir, subgraph_ids=None, sample_n=None, embed_component_ids=False):
    """
    Export the full bipartite graph (or a filtered/sampled version) to GraphML.
    - subgraph_ids: iterable of component IDs like ['C12','C7'] or integers [12,7]
    - sample_n: optional max number of edges to keep (after subgraph filtering)
    - embed_component_ids: include 'component_id' attribute on nodes
    """
    ensure_out(out_dir)
    try:
        import networkx as nx
    except Exception:
        return export_bipartite_csv(level_dir, out_dir, subgraph_ids=subgraph_ids, sample_n=sample_n,
                                    embed_component_ids=embed_component_ids)

    meta, comps, edges = load_level(level_dir)

    # Optional: filter to a subgraph consisting only of selected components
    if subgraph_ids:
        # Normalize to ints (strip any leading "C")
        norm = []
        for s in subgraph_ids:
            if isinstance(s, str) and s.upper().startswith('C'):
                s = s[1:]
            try:
                norm.append(int(s))
            except Exception:
                continue
        lmap, rmap = build_pair_to_component_maps(comps)
        edges = filter_edges_by_components(edges, lmap, rmap, keep_cis=norm)

    # Optional: sample edges for speed/scalability
    edges = sample_edges(edges, sample_n)

    # Build graph
    import networkx as nx
    G = nx.Graph()

    # Optional node attribute: component_id (only if we can map from pairs → component)
    comp_attr = {}
    if embed_component_ids:
        lmap, rmap = build_pair_to_component_maps(comps)
        comp_attr = {'L': lmap, 'R': rmap}  # lookups by ('L',i,j) or ('R',k,l)

    for (Li, Lj, Rk, Rl) in edges:
        Li, Lj, Rk, Rl = int(Li), int(Lj), int(Rk), int(Rl)
        ln = f"L_{Li}_{Lj}"
        rn = f"R_{Rk}_{Rl}"

        if ln not in G:
            attrs = {'side': 'L', 'i': Li, 'j': Lj}
            if embed_component_ids:
                attrs['component_id'] = comp_attr['L'].get(('L', Li, Lj), None)
            G.add_node(ln, **attrs)
        if rn not in G:
            attrs = {'side': 'R', 'k': Rk, 'l': Rl}
            if embed_component_ids:
                attrs['component_id'] = comp_attr['R'].get(('R', Rk, Rl), None)
            G.add_node(rn, **attrs)

        G.add_edge(ln, rn)

    # Write GraphML
    path = os.path.join(out_dir, 'bipartite.graphml')
    try:
        nx.write_graphml(G, path)
        return path
    except Exception:
        # Fallback to CSV
        return export_bipartite_csv(level_dir, out_dir, subgraph_ids=subgraph_ids, sample_n=sample_n,
                                    embed_component_ids=embed_component_ids)


def export_bipartite_csv(level_dir, out_dir, subgraph_ids=None, sample_n=None, embed_component_ids=False):
    """
    CSV fallback with optional subgraph and sampling.
    """
    ensure_out(out_dir)
    meta, comps, edges = load_level(level_dir)

    # Filter by components if requested
    if subgraph_ids:
        norm = []
        for s in subgraph_ids:
            if isinstance(s, str) and s.upper().startswith('C'):
                s = s[1:]
            try:
                norm.append(int(s))
            except Exception:
                continue
        lmap, rmap = build_pair_to_component_maps(comps)
        edges = filter_edges_by_components(edges, lmap, rmap, keep_cis=norm)

    # Sample edges if requested
    edges = sample_edges(edges, sample_n)

    # Optional: add component_id attributes
    lmap, rmap = ({} , {})
    if embed_component_ids:
        lmap, rmap = build_pair_to_component_maps(comps)

    nodes = {}
    for (Li, Lj, Rk, Rl) in edges:
        Li, Lj, Rk, Rl = int(Li), int(Lj), int(Rk), int(Rl)
        nodes[("L", Li, Lj)] = None
        nodes[("R", Rk, Rl)] = None

    nodes_csv = os.path.join(out_dir, 'bipartite_nodes.csv')
    edges_csv = os.path.join(out_dir, 'bipartite_edges.csv')

    with open(nodes_csv, 'w', encoding='utf-8') as f:
        # If embed_component_ids, write an extra column
        if embed_component_ids:
            f.write('id,side,i,j,k,l,component_id\n')
        else:
            f.write('id,side,i,j,k,l\n')

        for (side, a, b) in nodes:
            if side == 'L':
                comp_id = lmap.get(('L', a, b), None) if embed_component_ids else None
                if embed_component_ids:
                    f.write(f"L_{a}_{b},L,{a},{b},,,{comp_id}\n")
                else:
                    f.write(f"L_{a}_{b},L,{a},{b},,,\n")
            else:
                comp_id = rmap.get(('R', a, b), None) if embed_component_ids else None
                if embed_component_ids:
                    f.write(f"R_{a}_{b},R,,,{a},{b},{comp_id}\n")
                else:
                    f.write(f"R_{a}_{b},R,,,{a},{b}\n")

    with open(edges_csv, 'w', encoding='utf-8') as f:
        f.write('source,target\n')
        for (Li, Lj, Rk, Rl) in edges:
            f.write(f"L_{int(Li)}_{int(Lj)},R_{int(Rk)}_{int(Rl)}\n")

    return nodes_csv


# ---------------------------
#             CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('level_dir', help='Path to level_XXD directory')
    ap.add_argument('--out', required=True, help='Output directory for files')
    ap.add_argument('--graphml', nargs='*', default=['bipartite', 'components'],
                    choices=['bipartite', 'components'], help='Which artifacts to generate')
    ap.add_argument('--connect-by', default=None, choices=[None, 'signature', 'deg-profile'],
                    help='For components.graphml: connect components by signature or degree profile')
    ap.add_argument('--subgraph', nargs='*', default=None,
                    help='List of component IDs to keep in the bipartite export (e.g., C12 C7 18)')
    ap.add_argument('--sample', type=int, default=None,
                    help='If set, randomly sample up to N edges for bipartite export (after subgraph filtering)')
    ap.add_argument('--embed-component-ids', action='store_true',
                    help='Embed component_id attribute on bipartite nodes (fast; uses components.json mapping)')
    args = ap.parse_args()

    ensure_out(args.out)
    outputs = []

    if 'bipartite' in args.graphml:
        outputs.append(
            export_bipartite_graphml(
                args.level_dir, args.out,
                subgraph_ids=args.subgraph,
                sample_n=args.sample,
                embed_component_ids=args.embed_component_ids
            )
        )

    if 'components' in args.graphml:
        outputs.append(
            export_components_graphml(
                args.level_dir, args.out,
                connect_by=args.connect_by
            )
        )

    print("Generated:")
    for p in outputs:
        print(" ", p)


if __name__ == '__main__':
    main()