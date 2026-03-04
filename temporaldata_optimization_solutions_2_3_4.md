# TemporalData Performance Plan: Solutions 2, 3, and 4

This document describes how to implement three targeted optimizations in `temporaldata` to reduce `Data.slice()` overhead seen in profiling:

- **Solution 2**: Replace `Interval.__and__` sweep-line Python loop with a vectorized/binary-search implementation.
- **Solution 3**: Reduce `LazyInterval` attribute interception overhead by caching/materializing hot fields.
- **Solution 4**: Avoid repeated key-list reconstruction in `ArrayDict.keys()` by caching non-private keys.

The goal is to preserve behavior while removing Python-level overhead in the hot path.

---

## Baseline Problem

Current hot path:

- `Data.slice()` performs:
  - `out._domain = copy.copy(self._domain) & Interval(start, end)`
- `Interval.__and__` currently:
  - Iterates one boundary event at a time via `sorted_traversal(...)`
  - Uses `np.append(...)` inside the loop (reallocates repeatedly)
- On `LazyInterval`, repeated access to `.start`, `.end`, `keys()`, and `__len__` goes through `__getattribute__`, with frequent metadata checks.

This causes avoidable overhead, especially when domains have many disjoint segments.

---

## Solution 2: Rework `Interval.__and__` (Vectorized + Binary Search)

### Why this helps

The current implementation is Python-loop heavy and allocation-heavy. In practice, `Data.slice()` often intersects with a single interval, but even general intersections can be much faster if handled with:

- `np.searchsorted(...)` to locate overlap windows
- slicing and clipping in NumPy
- list accumulation + `np.concatenate(...)` once

### Current behavior constraints to preserve

`Interval.__and__` must continue to:

- require both operands to be sorted and disjoint (or raise)
- return only `start`/`end` in the resulting `Interval`
- drop zero-length intersections (`start == end`)
- keep output sorted and disjoint

### Recommended implementation approach

Implement a two-path strategy:

1. **Fast path for single-interval RHS (`len(other) == 1`)**
   - `idx_l = np.searchsorted(self.end, other_start, side="right")`
   - `idx_r = np.searchsorted(self.start, other_end, side="left")`
   - Slice candidate intervals once
   - Clip first/last boundaries with `max/min`
   - Filter out non-positive durations

2. **General path for multi-interval RHS**
   - Loop over each interval in `other` (small Python loop over intervals, not boundaries)
   - For each interval, locate candidate overlap block with `searchsorted`
   - Clip candidate block vectorially
   - Append resulting arrays to Python lists
   - `np.concatenate(...)` at the end

This avoids boundary-level generator overhead and avoids repeated `np.append`.

### Pseudocode (behavioral template)

```python
def __and__(self, other):
    validate_disjoint_and_sorted(self, other)

    if len(self) == 0 or len(other) == 0:
        return Interval(start=np.array([], dtype=np.float64),
                        end=np.array([], dtype=np.float64))

    out_starts = []
    out_ends = []

    for j in range(len(other)):
        a = other.start[j]
        b = other.end[j]

        left = np.searchsorted(self.end, a, side="right")
        right = np.searchsorted(self.start, b, side="left")

        if left >= right:
            continue

        s = self.start[left:right].copy()
        e = self.end[left:right].copy()

        s[0] = max(s[0], a)
        e[-1] = min(e[-1], b)

        keep = s < e
        if np.any(keep):
            out_starts.append(s[keep])
            out_ends.append(e[keep])

    if not out_starts:
        return Interval(start=np.array([], dtype=np.float64),
                        end=np.array([], dtype=np.float64))

    return Interval(
        start=np.concatenate(out_starts),
        end=np.concatenate(out_ends),
    )
```

### Edge cases checklist

- one or both intervals empty
- touching boundaries (`end == start`) should not create zero-length outputs
- all overlaps filtered out after clipping
- very large intervals arrays
- non-float inputs that are converted upstream

### Validation tests to add

- intersection with single interval
- intersection with multiple disjoint intervals
- no overlap case
- exact boundary-touch case
- random fuzz test vs current implementation (temporarily keep old impl as reference in tests)

---

## Solution 3: Reduce `LazyInterval` Access Overhead

### Why this helps

`LazyInterval.__getattribute__` currently intercepts almost every field access, and each access can call `keys()` and perform lazy-op checks. This is expensive in tight loops (`is_sorted`, `is_disjoint`, `__len__`, repeated `start/end` access).

### Recommended tactics

#### 3.1 Add an explicit lightweight accessor path for hot fields

Create helper methods that bypass generic key filtering:

- `_get_start_array()`
- `_get_end_array()`

Each helper:

- resolves pending lazy slice only if needed
- materializes `h5py.Dataset` to `np.ndarray` once
- stores materialized value back in `__dict__`
- returns array directly

Then update internals (`_maybe_first_dim`, `is_sorted`, `is_disjoint`, `slice` internals) to use helpers instead of generic `self.start` / `self.end` inside hot loops.

#### 3.2 Materialize eagerly when entering compute-heavy methods

For methods that repeatedly touch boundaries (`is_sorted`, `is_disjoint`, `__and__`, maybe `sort`), do:

- `start = self._get_start_array()`
- `end = self._get_end_array()`

and use local variables throughout.

This converts many dynamic attribute hits into local NumPy array reads.

#### 3.3 Keep class demotion logic, but avoid checking `all_loaded` too often

Today, each attribute access checks whether all keys are loaded and then mutates class from `LazyInterval` to `Interval`.

To reduce overhead:

- perform `all_loaded` checks only after a materialization event (not every read)
- optionally track a counter/flag of unresolved lazy arrays instead of scanning all keys each time

### Safety constraints

- Preserve lazy semantics for non-accessed attributes
- Preserve unicode conversion for keys in `_unicode_keys`
- Preserve current class demotion behavior once fully materialized

### Tests to add

- lazy interval remains lazy when untouched
- accessing only `start` does not force unrelated field materialization
- repeated `start/end` access does not rematerialize
- demotion to `Interval` still happens when expected

---

## Solution 4: Cache `ArrayDict.keys()` Results

### Why this helps

Current `keys()` reconstructs the list every time using:

- `filter(lambda x: not x.startswith("_"), self.__dict__)`

This is called often by `__getattribute__`, `__len__`, and other internals. Caching avoids repeated string filtering and lambda calls.

### Recommended implementation

#### 4.1 Add a private cache field

- `_cached_public_keys: list[str] | None = None`

#### 4.2 Invalidate cache on any attribute mutation

In `ArrayDict.__setattr__`:

- call existing validation logic
- set `_cached_public_keys = None` for non-private key writes

If keys can be deleted anywhere in code, add matching invalidation in `__delattr__` as well.

#### 4.3 Use cached list in `keys()`

`keys()` logic:

- if cache exists, return it (or a tuple/list copy, depending on mutability preference)
- otherwise build once from `self.__dict__`, store cache, return

### Design choice: return cached list directly vs copy

Recommended:

- store as tuple internally
- return `list(tuple_cache)` from `keys()`

This prevents accidental external mutation from corrupting cache state.

### Compatibility concerns

- Ensure all existing code expecting list semantics still works
- Ensure private fields prefixed `_` stay excluded
- Ensure lazy class transitions and dynamic attribute additions still invalidate cache correctly

### Tests to add

- `keys()` returns same public keys before/after non-key attribute updates
- adding new public attribute invalidates cache and appears in results
- private attribute additions do not appear in keys

---

## Recommended Implementation Order

1. **Solution 4 first** (low risk, broad overhead reduction)
2. **Solution 3 next** (reduce hot-path lazy overhead)
3. **Solution 2 last** (highest impact, most behavior-sensitive)

This order simplifies debugging and makes perf deltas easier to attribute.

---

## Benchmarking Plan

Run micro-benchmarks before/after each step.

### Bench 1: `Interval.__and__` throughput

- synthetic `Interval` with `N={10,100,1000,5000}` disjoint segments
- intersect with:
  - one interval
  - 10 intervals
- measure median runtime over many repeats

### Bench 2: `Data.slice` latency

- representative AJILE recording
- repeated random slices (same distribution as training sampler)
- track median and p95

### Bench 3: End-to-end dataloader batch time

- `num_workers=0`, fixed seed
- measure dataloader iteration latency only
- compare baseline and each optimization stage

---

## Definition of Done

- unit tests pass for interval behavior and lazy semantics
- profile shows `temporaldata/interval.py(...): __and__` no longer dominating sample time
- no regression in correctness on existing training pipeline
- benchmark reports attached to PR or notes

