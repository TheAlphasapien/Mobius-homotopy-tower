"""
cd_tower.py - Core module for Cayley-Dickson tower computation

Provides:
  - CayleyDickson algebra class
  - CDLevel: computed structure of one tower level, saveable/loadable
  - Incremental computation using previous level's multiplication table
  
Usage:
  # Compute from scratch
  level = CDLevel.compute(dim=32)
  level.save("level_32D")
  
  # Load and continue
  prev = CDLevel.load("level_32D")
  next_level = CDLevel.compute(dim=64, prev_table=prev.mult_table)
  next_level.save("level_64D")

Author: Adam Kevin Morgan / Boundary Paradigm Program
"""

import numpy as np
from collections import defaultdict
import time
import os
import json


class CayleyDickson:
    """Cayley-Dickson algebra of arbitrary dimension."""
    
    def __init__(self, dim):
        self.dim = dim
        if dim == 1:
            self.half = None
        else:
            self.half = CayleyDickson(dim // 2)
    
    def multiply(self, x, y):
        if self.dim == 1:
            return x * y
        n = self.dim // 2
        a, b = x[:n], x[n:]
        c, d = y[:n], y[n:]
        d_conj = self.half.conjugate(d)
        c_conj = self.half.conjugate(c)
        return np.concatenate([
            self.half.multiply(a, c) - self.half.multiply(d_conj, b),
            self.half.multiply(d, a) + self.half.multiply(b, c_conj)])
    
    def conjugate(self, x):
        if self.dim == 1:
            return x.copy()
        n = self.dim // 2
        a, b = x[:n], x[n:]
        return np.concatenate([self.half.conjugate(a), -b])


def compute_mult_table(dim, prev_table=None, report_fn=None):
    """
    Compute the DIM x DIM x DIM multiplication table M[i,j,k] = (e_i * e_j)[k].
    
    If prev_table is provided (shape [dim/2, dim/2, dim/2]), uses it to
    accelerate computation of the new level.
    
    Returns numpy array of shape (dim, dim, dim).
    """
    if report_fn is None:
        report_fn = lambda msg: print(msg, flush=True)
    
    basis = np.eye(dim)
    
    if prev_table is not None and prev_table.shape[0] == dim // 2:
        # Incremental: use CD formula to build new table from previous
        n = dim // 2
        M = np.zeros((dim, dim, dim))
        report_fn(f"  Building {dim}x{dim} table incrementally from {n}x{n} table...")
        
        # For basis elements in the lower half (0..n-1), products are known
        # from prev_table. For mixed and upper half products, use CD formula.
        
        # We still need the algebra object for the mixed products
        A = CayleyDickson(dim)
        
        t0 = time.time()
        for i in range(dim):
            if i % max(1, dim // 8) == 0:
                elapsed = time.time() - t0
                report_fn(f"    Row {i}/{dim} ({elapsed:.1f}s)")
            
            for j in range(dim):
                if i < n and j < n:
                    # Both in lower half: embed previous result
                    M[i, j, :n] = prev_table[i, j, :]
                    # Upper half is zero for pure lower-half products
                else:
                    M[i, j, :] = A.multiply(basis[i], basis[j])
        
        return M
    
    else:
        # From scratch
        A = CayleyDickson(dim)
        M = np.zeros((dim, dim, dim))
        report_fn(f"  Building {dim}x{dim} table from scratch...")
        
        t0 = time.time()
        for i in range(dim):
            if i % max(1, dim // 8) == 0:
                elapsed = time.time() - t0
                report_fn(f"    Row {i}/{dim} ({elapsed:.1f}s)")
            for j in range(dim):
                M[i, j, :] = A.multiply(basis[i], basis[j])
        
        return M


def find_zero_divisors(M, dim, report_fn=None):
    """
    Find all zero divisor pairs (e_i + e_j)(e_k + e_l) = 0
    using the precomputed multiplication table.
    
    Returns list of tuples (i, j, k, l) with i < j and k < l.
    """
    if report_fn is None:
        report_fn = lambda msg: print(msg, flush=True)
    
    pair_indices = [(i, j) for i in range(dim) for j in range(i+1, dim)]
    n_pairs = len(pair_indices)
    
    report_fn(f"  Pairs: {n_pairs}, products to check: {n_pairs**2:,}")
    
    zd = []
    t0 = time.time()
    last_report = time.time()
    
    for li, (i, j) in enumerate(pair_indices):
        Mij = M[i, :, :] + M[j, :, :]
        
        for ri, (k, l) in enumerate(pair_indices):
            prod = Mij[k, :] + Mij[l, :]
            if np.dot(prod, prod) < 1e-10:
                zd.append((i, j, k, l))
        
        if time.time() - last_report > 30:
            elapsed = time.time() - t0
            frac = (li + 1) / n_pairs
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            report_fn(f"    {li+1}/{n_pairs} ({frac*100:.1f}%), "
                      f"{len(zd)} ZDs, ETA: {eta:.0f}s")
            last_report = time.time()
    
    return zd


def build_graph_and_components(zd_pairs):
    """
    Build bipartite ZD graph and find connected components.
    
    Returns:
      - undirected_adj: dict mapping node -> set of neighbors
      - components: list of sets of nodes, sorted by size descending
      - node info for each component
    """
    undirected_adj = defaultdict(set)
    for (i, j, k, l) in zd_pairs:
        left = ('L', i, j)
        right = ('R', k, l)
        undirected_adj[left].add(right)
        undirected_adj[right].add(left)
    
    all_nodes = set(undirected_adj.keys())
    
    visited = set()
    components = []
    for start in sorted(all_nodes):
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for nb in undirected_adj[node]:
                if nb not in visited:
                    queue.append(nb)
        components.append(comp)
    
    components.sort(key=len, reverse=True)
    
    return dict(undirected_adj), components


def analyze_components(components, undirected_adj, dim):
    """
    Compute per-component metadata: size, degree distribution,
    index ranges, octonionic omissions, etc.
    
    Returns list of dicts, one per component.
    """
    results = []
    
    for ci, comp in enumerate(components):
        left_nodes = sorted([n for n in comp if n[0] == 'L'])
        right_nodes = sorted([n for n in comp if n[0] == 'R'])
        
        # Degree counts (left side)
        deg_counts = defaultdict(int)
        for n in left_nodes:
            deg_counts[len(undirected_adj[n])] += 1
        
        # All basis indices used
        all_idx = set()
        for n in comp:
            all_idx.add(n[1])
            all_idx.add(n[2])
        
        # Octonionic analysis
        oct_idx = set(x for x in all_idx if x < 8)
        missing_oct = sorted(set(range(1, 8)) - oct_idx)
        
        # Boundary analysis (low half vs high half)
        half = dim // 2
        n_low = len([x for x in all_idx if x < half])
        n_high = len([x for x in all_idx if x >= half])
        
        # Left/right pair sets (for chirality testing)
        left_pairs = frozenset((n[1], n[2]) for n in left_nodes)
        right_pairs = frozenset((n[1], n[2]) for n in right_nodes)
        
        # Signature for type classification
        sig = (len(comp), tuple(sorted(deg_counts.items())))
        
        results.append({
            'ci': ci,
            'size': len(comp),
            'n_left': len(left_nodes),
            'n_right': len(right_nodes),
            'deg_counts': dict(sorted(deg_counts.items())),
            'missing_oct': missing_oct,
            'n_low': n_low,
            'n_high': n_high,
            'left_pairs': left_pairs,
            'right_pairs': right_pairs,
            'sig': sig,
            'all_indices': sorted(all_idx),
        })
    
    return results


class CDLevel:
    """
    Complete computed structure of one Cayley-Dickson tower level.
    
    Attributes:
      dim: algebra dimension
      mult_table: numpy array shape (dim, dim, dim)
      zd_pairs: list of (i, j, k, l) tuples
      undirected_adj: dict mapping node -> set of neighbors
      components: list of sets (connected components)
      comp_info: list of dicts (per-component metadata)
      degree_dist: dict mapping degree -> count
      timings: dict of computation times
    """
    
    def __init__(self):
        self.dim = None
        self.mult_table = None
        self.zd_pairs = None
        self.undirected_adj = None
        self.components = None
        self.comp_info = None
        self.degree_dist = None
        self.timings = {}
    
    @classmethod
    def compute(cls, dim, prev_table=None, report_fn=None):
        """
        Compute the full structure for a given dimension.
        
        Args:
          dim: algebra dimension (must be power of 2, >= 16)
          prev_table: optional multiplication table from previous level
          report_fn: callback for progress messages
        """
        if report_fn is None:
            report_fn = lambda msg: print(msg, flush=True)
        
        level = cls()
        level.dim = dim
        
        report_fn(f"\n{'='*60}")
        report_fn(f"COMPUTING {dim}D LEVEL")
        report_fn(f"{'='*60}")
        
        # Step 1: Multiplication table
        report_fn(f"\nStep 1: Multiplication table")
        t0 = time.time()
        level.mult_table = compute_mult_table(dim, prev_table, report_fn)
        level.timings['mult_table'] = time.time() - t0
        report_fn(f"  Done in {level.timings['mult_table']:.1f}s")
        
        # Verify
        assert abs(level.mult_table[1, 1, 0] + 1.0) < 1e-10, "e1^2 != -1"
        
        # Step 2: Zero divisor search
        report_fn(f"\nStep 2: Zero divisor search")
        t0 = time.time()
        level.zd_pairs = find_zero_divisors(level.mult_table, dim, report_fn)
        level.timings['zd_search'] = time.time() - t0
        report_fn(f"  Found {len(level.zd_pairs)} ZD pairs in {level.timings['zd_search']:.1f}s")
        
        # Step 3: Graph and components
        report_fn(f"\nStep 3: Graph and components")
        t0 = time.time()
        level.undirected_adj, level.components = build_graph_and_components(level.zd_pairs)
        level.timings['graph'] = time.time() - t0
        report_fn(f"  {len(level.components)} components in {level.timings['graph']:.1f}s")
        
        # Step 4: Component analysis
        report_fn(f"\nStep 4: Component analysis")
        t0 = time.time()
        level.comp_info = analyze_components(
            level.components, level.undirected_adj, dim)
        level.timings['analysis'] = time.time() - t0
        
        # Degree distribution
        level.degree_dist = defaultdict(int)
        for node, neighbors in level.undirected_adj.items():
            if node[0] == 'L':
                level.degree_dist[len(neighbors)] += 1
        level.degree_dist = dict(sorted(level.degree_dist.items()))
        
        total = sum(level.timings.values())
        report_fn(f"\n  Total time: {total:.1f}s")
        
        return level
    
    def save(self, directory):
        """
        Save level structure to directory.
        Creates: mult_table.npy, zd_pairs.npy, metadata.json, 
                 graph_adj.npz, components.json
        """
        os.makedirs(directory, exist_ok=True)
        
        # Multiplication table (largest file)
        np.save(os.path.join(directory, 'mult_table.npy'), self.mult_table)
        
        # ZD pairs as numpy array
        if self.zd_pairs:
            np.save(os.path.join(directory, 'zd_pairs.npy'),
                    np.array(self.zd_pairs, dtype=np.int32))
        
        # Graph adjacency as edge list
        edges = []
        for node, neighbors in self.undirected_adj.items():
            for nb in neighbors:
                # Only store L->R edges to avoid duplication
                if node[0] == 'L' and nb[0] == 'R':
                    edges.append([node[1], node[2], nb[1], nb[2]])
        np.save(os.path.join(directory, 'edges.npy'),
                np.array(edges, dtype=np.int32))
        
        # Metadata
        meta = {
            'dim': self.dim,
            'n_zd_pairs': len(self.zd_pairs),
            'n_components': len(self.components),
            'degree_dist': {str(k): v for k, v in self.degree_dist.items()},
            'n_strata': len(self.degree_dist),
            'timings': self.timings,
        }
        with open(os.path.join(directory, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Component info (serializable subset)
        comp_data = []
        for ci in self.comp_info:
            comp_data.append({
                'ci': ci['ci'],
                'size': ci['size'],
                'n_left': ci['n_left'],
                'n_right': ci['n_right'],
                'deg_counts': ci['deg_counts'],
                'missing_oct': ci['missing_oct'],
                'n_low': ci['n_low'],
                'n_high': ci['n_high'],
                'sig': [ci['sig'][0], ci['sig'][1]],
                'left_pairs': sorted(ci['left_pairs']),
                'right_pairs': sorted(ci['right_pairs']),
            })
        with open(os.path.join(directory, 'components.json'), 'w') as f:
            json.dump(comp_data, f, indent=1)
        
        print(f"Saved {self.dim}D level to {directory}/")
    
    @classmethod
    def load(cls, directory):
        """
        Load level structure from directory.
        """
        level = cls()
        
        # Metadata first
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        level.dim = meta['dim']
        level.degree_dist = {int(k): v for k, v in meta['degree_dist'].items()}
        level.timings = meta.get('timings', {})
        
        # Multiplication table
        level.mult_table = np.load(os.path.join(directory, 'mult_table.npy'))
        
        # ZD pairs
        zd_arr = np.load(os.path.join(directory, 'zd_pairs.npy'))
        level.zd_pairs = [tuple(row) for row in zd_arr]
        
        # Rebuild graph from edge list
        edges = np.load(os.path.join(directory, 'edges.npy'))
        level.undirected_adj = defaultdict(set)
        for row in edges:
            left = ('L', int(row[0]), int(row[1]))
            right = ('R', int(row[2]), int(row[3]))
            level.undirected_adj[left].add(right)
            level.undirected_adj[right].add(left)
        level.undirected_adj = dict(level.undirected_adj)
        
        # Rebuild components from graph
        all_nodes = set(level.undirected_adj.keys())
        visited = set()
        level.components = []
        for start in sorted(all_nodes):
            if start in visited:
                continue
            comp = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                for nb in level.undirected_adj.get(node, set()):
                    if nb not in visited:
                        queue.append(nb)
            level.components.append(comp)
        level.components.sort(key=len, reverse=True)
        
        # Component info
        with open(os.path.join(directory, 'components.json'), 'r') as f:
            comp_data = json.load(f)
        level.comp_info = []
        for cd in comp_data:
            cd['left_pairs'] = frozenset(tuple(p) for p in cd['left_pairs'])
            cd['right_pairs'] = frozenset(tuple(p) for p in cd['right_pairs'])
            cd['sig'] = (cd['sig'][0], tuple(tuple(x) for x in cd['sig'][1]))
            cd['deg_counts'] = {int(k): v for k, v in cd['deg_counts'].items()}
            level.comp_info.append(cd)
        
        print(f"Loaded {level.dim}D level from {directory}/ "
              f"({len(level.zd_pairs)} ZD pairs, {len(level.components)} components)")
        
        return level
    
    def summary(self):
        """Print a concise summary of this level."""
        print(f"\n{'='*60}")
        print(f"{self.dim}D LEVEL SUMMARY")
        print(f"{'='*60}")
        print(f"  ZD pairs: {len(self.zd_pairs)}")
        print(f"  Components: {len(self.components)}")
        print(f"  Strata: {len(self.degree_dist)}")
        print(f"  Degrees: {sorted(self.degree_dist.keys())}")
        print(f"\n  Size distribution:")
        size_dist = defaultdict(int)
        for comp in self.components:
            size_dist[len(comp)] += 1
        for size, count in sorted(size_dist.items(), reverse=True):
            print(f"    Size {size}: {count}")
        print(f"\n  Degree distribution:")
        for deg, count in sorted(self.degree_dist.items()):
            print(f"    Degree {deg}: {count} nodes (kernel dim {2*deg})")
        print(f"\n  Component types:")
        type_counts = defaultdict(int)
        for ci in self.comp_info:
            type_counts[ci['sig']] += 1
        for sig, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            size, degs = sig
            print(f"    {count:>3d}x size={size}, degs={dict(degs)}")
