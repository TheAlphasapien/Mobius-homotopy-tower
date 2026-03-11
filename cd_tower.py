"""
cd_tower.py - Core module for Cayley-Dickson tower computation

Adds:
 - Parallel ZD search (shared memory)
 - Checkpoint/resume per (i,j) block
 - Optional cheap pruning (safe bound)
"""

import numpy as np
from collections import defaultdict
import time
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

# ---------------------------
#   Cayley–Dickson algebra
# ---------------------------

class CayleyDickson:
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
            self.half.multiply(d, a) + self.half.multiply(b, c_conj)
        ])

    def conjugate(self, x):
        if self.dim == 1:
            return x.copy()
        n = self.dim // 2
        a, b = x[:n], x[n:]
        return np.concatenate([self.half.conjugate(a), -b])


# -------------------------------------------
#   Multiplication table (parallel optional)
# -------------------------------------------

def _compute_block(dim, prev_table, start_row, end_row):
    basis = np.eye(dim)
    A = CayleyDickson(dim)
    rows = end_row - start_row
    block = np.zeros((rows, dim, dim), dtype=np.float64)

    if prev_table is not None and prev_table.shape[0] == dim // 2:
        n = dim // 2
        for i in range(start_row, end_row):
            bi = i - start_row
            for j in range(dim):
                if i < n and j < n:
                    block[bi, j, :n] = prev_table[i, j, :]
                else:
                    block[bi, j, :] = A.multiply(basis[i], basis[j])
    else:
        for i in range(start_row, end_row):
            bi = i - start_row
            for j in range(dim):
                block[bi, j, :] = A.multiply(basis[i], basis[j])

    return (start_row, block)


def compute_mult_table(dim, prev_table=None, report_fn=None, workers=None):
    if report_fn is None:
        report_fn = lambda msg: print(msg, flush=True)

    if not workers or workers <= 1:
        basis = np.eye(dim)
        A = CayleyDickson(dim)
        M = np.zeros((dim, dim, dim), dtype=np.float64)
        if prev_table is not None and prev_table.shape[0] == dim // 2:
            n = dim // 2
            report_fn(f" Building {dim}x{dim}x{dim} incrementally from {n}x{n}x{n} (serial)...")
        else:
            report_fn(f" Building {dim}x{dim}x{dim} from scratch (serial)...")

        t0 = time.time()
        for i in range(dim):
            if i % max(1, dim // 8) == 0:
                elapsed = time.time() - t0
                report_fn(f" Row {i}/{dim} ({elapsed:.1f}s)")
            for j in range(dim):
                if prev_table is not None and prev_table.shape[0] == dim // 2 and i < dim // 2 and j < dim // 2:
                    M[i, j, :dim//2] = prev_table[i, j, :]
                else:
                    M[i, j, :] = A.multiply(basis[i], basis[j])
        return M

    # Parallel path
    W = workers if workers > 0 else (os.cpu_count() or 1)
    block_rows = max(1, dim // (4 * W))
    ranges = []
    s = 0
    while s < dim:
        e = min(dim, s + block_rows)
        ranges.append((s, e))
        s = e

    if prev_table is not None and prev_table.shape[0] == dim // 2:
        n = dim // 2
        report_fn(f" Building {dim}x{dim}x{dim} incrementally from {n}x{n}x{n} with {W} workers...")
    else:
        report_fn(f" Building {dim}x{dim}x{dim} from scratch with {W} workers...")

    t0 = time.time()
    M = np.zeros((dim, dim, dim), dtype=np.float64)
    with ProcessPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(_compute_block, dim, prev_table, s, e) for (s, e) in ranges]
        completed = 0
        for fut in as_completed(futs):
            s0, block = fut.result()
            r = block.shape[0]
            M[s0:s0 + r, :, :] = block
            completed += r
            if completed % max(1, dim // 8) == 0 or completed == dim:
                elapsed = time.time() - t0
                report_fn(f" Rows {completed}/{dim} ({elapsed:.1f}s)")

    return M


# ---------------------------------
#   Zero-divisors (parallel + ckpt)
# ---------------------------------

def _zd_worker(shm_name, shape, dtype_str, li_start, li_end, pair_indices, eps, use_prune):
    shm = SharedMemory(name=shm_name)
    try:
        M = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        dim = shape[0]
        results = []
        pairs = np.array(pair_indices, dtype=np.int32)
        p0 = pairs[:, 0]
        p1 = pairs[:, 1]
        sqrt_eps = np.sqrt(eps) if use_prune else None
        for li in range(li_start, li_end):
            i, j = pair_indices[li]
            Mij = M[i, :, :] + M[j, :, :]
            # cheap pruning (safe): | ||rowk|| - ||rowl|| | < sqrt_eps must hold for any zero
            if use_prune:
                row_norms = np.sqrt(np.einsum('ij,ij->i', Mij, Mij))  # (dim,)
                diff = np.abs(row_norms[p0] - row_norms[p1])          # (n_pairs,)
                candidate_idx = np.nonzero(diff < sqrt_eps)[0]
                if candidate_idx.size == 0:
                    continue
                c0 = p0[candidate_idx]
                c1 = p1[candidate_idx]
                S = Mij[c0, :] + Mij[c1, :]
                norms = np.einsum('ij,ij->i', S, S)
                mask = norms < eps
                if np.any(mask):
                    idxs = candidate_idx[np.nonzero(mask)[0]]
                    for t in idxs:
                        results.append((i, j, int(p0[t]), int(p1[t])))
            else:
                S = Mij[p0, :] + Mij[p1, :]
                norms = np.einsum('ij,ij->i', S, S)
                mask = norms < eps
                if np.any(mask):
                    idxs = np.nonzero(mask)[0]
                    for t in idxs:
                        results.append((i, j, int(p0[t]), int(p1[t])))
        return results
    finally:
        shm.close()


def _load_progress(checkpoint_dir):
    prog_path = os.path.join(checkpoint_dir, 'zd_progress.json')
    if os.path.exists(prog_path):
        with open(prog_path, 'r') as f:
            return json.load(f)
    return None


def _save_progress(checkpoint_dir, progress):
    os.makedirs(checkpoint_dir, exist_ok=True)
    tmp = os.path.join(checkpoint_dir, 'zd_progress.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, os.path.join(checkpoint_dir, 'zd_progress.json'))


def _block_filename(checkpoint_dir, s, e):
    return os.path.join(checkpoint_dir, f'zd_block_{s}_{e}.npy')


def find_zero_divisors(M, dim, report_fn=None, workers=None, eps=1e-10, checkpoint_dir=None, resume=False, prune=False):
    if report_fn is None:
        report_fn = lambda msg: print(msg, flush=True)

    pair_indices = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    n_pairs = len(pair_indices)
    report_fn(f" Pairs: {n_pairs}, products to check: {n_pairs**2:,}")

    # Determine block plan (for both serial and parallel, so checkpoint is consistent)
    W = workers if (workers and workers > 0) else (os.cpu_count() or 1)
    block_size = max(1, n_pairs // (W * 4))
    ranges = []
    s = 0
    while s < n_pairs:
        e = min(n_pairs, s + block_size)
        ranges.append((s, e))
        s = e

    # Initialize or load progress
    progress = None
    done_blocks = set()
    if checkpoint_dir:
        progress = _load_progress(checkpoint_dir)
        if progress and resume:
            # Sanity checks for compatibility
            if progress.get('dim') != dim or abs(progress.get('eps', eps) - eps) > 0:
                report_fn(" Progress file incompatible; ignoring resume.")
            else:
                block_size = progress.get('block_size', block_size)
                done_blocks = set(tuple(x) for x in progress.get('done_blocks', []))
                report_fn(f" Resuming ZD: {len(done_blocks)}/{len(ranges)} blocks already done")
        else:
            # fresh progress
            progress = {
                'dim': dim,
                'eps': eps,
                'block_size': block_size,
                'done_blocks': []
            }
            _save_progress(checkpoint_dir, progress)

    def mark_done(s, e):
        if checkpoint_dir:
            done_blocks.add((s, e))
            progress['done_blocks'] = list(sorted(list(done_blocks)))
            _save_progress(checkpoint_dir, progress)

    # Helper to compute one block (serial path)
    def compute_block_serial(s, e):
        # Attach shared mem path by reusing worker function with one block via local shm
        dtype_str = str(M.dtype)
        shm = SharedMemory(create=True, size=M.nbytes)
        try:
            M_sh = np.ndarray(M.shape, dtype=M.dtype, buffer=shm.buf)
            M_sh[...] = M
            part = _zd_worker(shm.name, M.shape, dtype_str, s, e, pair_indices, eps, prune)
        finally:
            shm.close(); shm.unlink()
        return part

    # Decide serial vs parallel execution
    results_files = []
    t0 = time.time()
    if not workers or workers <= 1:
        for (s, e) in ranges:
            if checkpoint_dir:
                blk_path = _block_filename(checkpoint_dir, s, e)
                if (s, e) in done_blocks or os.path.exists(blk_path):
                    results_files.append(blk_path)
                    continue
            part = compute_block_serial(s, e)
            if checkpoint_dir:
                blk_path = _block_filename(checkpoint_dir, s, e)
                np.save(blk_path, np.array(part, dtype=np.int32))
                results_files.append(blk_path)
                mark_done(s, e)
            else:
                results_files.append(part)  # store in-memory list
            if (len(results_files) % max(1, len(ranges)//8)) == 0:
                elapsed = time.time() - t0
                report_fn(f" ZD blocks {len(results_files)}/{len(ranges)} ({elapsed:.1f}s)")
    else:
        # Parallel: put M into shared memory once
        ctx = get_context("spawn")
        dtype_str = str(M.dtype)
        shm = SharedMemory(create=True, size=M.nbytes)
        try:
            M_sh = np.ndarray(M.shape, dtype=M.dtype, buffer=shm.buf)
            M_sh[...] = M
            with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
                futs = []
                # Submit only blocks not already done
                pending_blocks = []
                for (s, e) in ranges:
                    blk_path = _block_filename(checkpoint_dir, s, e) if checkpoint_dir else None
                    if checkpoint_dir and ((s, e) in done_blocks or (blk_path and os.path.exists(blk_path))):
                        results_files.append(blk_path)
                        continue
                    fut = ex.submit(_zd_worker, shm.name, M.shape, dtype_str, s, e, pair_indices, eps, prune)
                    futs.append((fut, s, e, blk_path))
                    pending_blocks.append((s, e))
                completed = 0
                for fut, s, e, blk_path in futs:
                    part = fut.result()
                    if checkpoint_dir:
                        np.save(blk_path, np.array(part, dtype=np.int32))
                        results_files.append(blk_path)
                        mark_done(s, e)
                    else:
                        results_files.append(part)
                    completed += 1
                    if completed % max(1, len(futs)//8) == 0 or completed == len(futs):
                        elapsed = time.time() - t0
                        report_fn(f" ZD blocks {completed}/{len(futs)} ({elapsed:.1f}s)")
        finally:
            shm.close(); shm.unlink()

    # Merge results
    if checkpoint_dir:
        pairs = []
        for rf in results_files:
            if rf and os.path.exists(rf):
                arr = np.load(rf)
                if arr.size:
                    pairs.append(arr)
        if pairs:
            out = np.vstack(pairs).tolist()
        else:
            out = []
        report_fn(f" ZD merge: {len(results_files)} blocks, total {len(out)} pairs")
        return out
    else:
        # results_files is a list of lists
        out = []
        for part in results_files:
            if isinstance(part, list):
                out.extend(part)
            elif isinstance(part, np.ndarray):
                out.extend(part.tolist())
        report_fn(f" ZD merge (memory): total {len(out)} pairs")
        return out


# -------------------------------
#       Component graph & stats
# -------------------------------

def build_graph_and_components(zd_pairs):
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
    results = []
    for ci, comp in enumerate(components):
        left_nodes = sorted([n for n in comp if n[0] == 'L'])
        right_nodes = sorted([n for n in comp if n[0] == 'R'])
        deg_counts = defaultdict(int)
        for n in left_nodes:
            deg_counts[len(undirected_adj[n])] += 1
        all_idx = set()
        for n in comp:
            all_idx.add(n[1]); all_idx.add(n[2])
        oct_idx = set(x for x in all_idx if x < 8)
        missing_oct = sorted(set(range(1, 8)) - oct_idx)
        half = dim // 2
        n_low = len([x for x in all_idx if x < half])
        n_high = len([x for x in all_idx if x >= half])
        left_pairs = frozenset((n[1], n[2]) for n in left_nodes)
        right_pairs = frozenset((n[1], n[2]) for n in right_nodes)
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
    def compute(cls, dim, prev_table=None, report_fn=None, workers=None, workers_zd=None,
                checkpoint_dir=None, resume=False, prune=False, eps=1e-10):
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
        level.mult_table = compute_mult_table(dim, prev_table, report_fn, workers=workers)
        level.timings['mult_table'] = time.time() - t0
        report_fn(f" Done in {level.timings['mult_table']:.1f}s")

        assert abs(level.mult_table[1, 1, 0] + 1.0) < 1e-10, "Sanity check failed: e1^2 != -1"

        # Step 2: Zero divisor search
        report_fn(f"\nStep 2: Zero divisor search")
        t0 = time.time()
        level.zd_pairs = find_zero_divisors(
            level.mult_table, dim, report_fn,
            workers=workers_zd, eps=eps,
            checkpoint_dir=checkpoint_dir, resume=resume, prune=prune
        )
        level.timings['zd_search'] = time.time() - t0
        report_fn(f" Found {len(level.zd_pairs)} ZD pairs in {level.timings['zd_search']:.1f}s")

        # Step 3: Graph and components
        report_fn(f"\nStep 3: Graph and components")
        t0 = time.time()
        level.undirected_adj, level.components = build_graph_and_components(level.zd_pairs)
        level.timings['graph'] = time.time() - t0
        report_fn(f" {len(level.components)} components in {level.timings['graph']:.1f}s")

        # Step 4: Component analysis
        report_fn(f"\nStep 4: Component analysis")
        t0 = time.time()
        level.comp_info = analyze_components(level.components, level.undirected_adj, dim)
        level.timings['analysis'] = time.time() - t0

        # Degree distribution
        level.degree_dist = defaultdict(int)
        for node, neighbors in level.undirected_adj.items():
            if node[0] == 'L':
                level.degree_dist[len(neighbors)] += 1
        level.degree_dist = dict(sorted(level.degree_dist.items()))

        total = sum(level.timings.values())
        report_fn(f"\n Total time: {total:.1f}s")

        return level

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'mult_table.npy'), self.mult_table)
        if self.zd_pairs:
            np.save(os.path.join(directory, 'zd_pairs.npy'), np.array(self.zd_pairs, dtype=np.int32))
        edges = []
        for node, neighbors in self.undirected_adj.items():
            for nb in neighbors:
                if node[0] == 'L' and nb[0] == 'R':
                    edges.append([node[1], node[2], nb[1], nb[2]])
        np.save(os.path.join(directory, 'edges.npy'), np.array(edges, dtype=np.int32))
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
        comp_data = []
        for ci in self.comp_info:
            comp_data.append({
                'ci': ci['ci'], 'size': ci['size'], 'n_left': ci['n_left'], 'n_right': ci['n_right'],
                'deg_counts': ci['deg_counts'], 'missing_oct': ci['missing_oct'],
                'n_low': ci['n_low'], 'n_high': ci['n_high'], 'sig': [ci['sig'][0], ci['sig'][1]],
                'left_pairs': sorted(ci['left_pairs']), 'right_pairs': sorted(ci['right_pairs']),
            })
        with open(os.path.join(directory, 'components.json'), 'w') as f:
            json.dump(comp_data, f, indent=1)
        print(f"Saved {self.dim}D level to {directory}/")

    @classmethod
    def load(cls, directory):
        level = cls()
        with open(os.path.join(directory, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        level.dim = meta['dim']
        level.degree_dist = {int(k): v for k, v in meta['degree_dist'].items()}
        level.timings = meta.get('timings', {})
        level.mult_table = np.load(os.path.join(directory, 'mult_table.npy'))
        zd_arr = np.load(os.path.join(directory, 'zd_pairs.npy'))
        level.zd_pairs = [tuple(row) for row in zd_arr]
        edges = np.load(os.path.join(directory, 'edges.npy'))
        level.undirected_adj = defaultdict(set)
        for row in edges:
            left = ('L', int(row[0]), int(row[1]))
            right = ('R', int(row[2]), int(row[3]))
            level.undirected_adj[left].add(right)
            level.undirected_adj[right].add(left)
        level.undirected_adj = dict(level.undirected_adj)
        all_nodes = set(level.undirected_adj.keys())
        visited = set()
        level.components = []
        for start in sorted(all_nodes):
            if start in visited:
                continue
            comp = set(); queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited: continue
                visited.add(node); comp.add(node)
                for nb in level.undirected_adj.get(node, set()):
                    if nb not in visited: queue.append(nb)
            level.components.append(comp)
        level.components.sort(key=len, reverse=True)
        with open(os.path.join(directory, 'components.json'), 'r') as f:
            comp_data = json.load(f)
        level.comp_info = []
        for cd in comp_data:
            cd['left_pairs'] = frozenset(tuple(p) for p in cd['left_pairs'])
            cd['right_pairs'] = frozenset(tuple(p) for p in cd['right_pairs'])
            cd['sig'] = (cd['sig'][0], tuple(tuple(x) for x in cd['sig'][1]))
            cd['deg_counts'] = {int(k): v for k, v in cd['deg_counts'].items()}
            level.comp_info.append(cd)
        print(f"Loaded {level.dim}D level from {directory}/ (" \
              f"{len(level.zd_pairs)} ZD pairs, {len(level.components)} components)")
        return level

    def summary(self):
        print(f"\n{'='*60}")
        print(f"{self.dim}D LEVEL SUMMARY")
        print(f"{'='*60}")
        print(f" ZD pairs: {len(self.zd_pairs)}")
        print(f" Components: {len(self.components)}")
        print(f" Strata: {len(self.degree_dist)}")
        print(f" Degrees: {sorted(self.degree_dist.keys())}")
        print(f"\n Size distribution:")
        size_dist = defaultdict(int)
        for comp in self.components:
            size_dist[len(comp)] += 1
        for size, count in sorted(size_dist.items(), reverse=True):
            print(f" Size {size}: {count}")
        print(f"\n Degree distribution:")
        for deg, count in sorted(self.degree_dist.items()):
            print(f" Degree {deg}: {count} nodes (kernel dim {2*deg})")
        print(f"\n Component types:")
        type_counts = defaultdict(int)
        for ci in self.comp_info:
            type_counts[ci['sig']] += 1
        for sig, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            size, degs = sig
            print(f" {count:>3d}x size={size}, degs={dict(degs)}")
