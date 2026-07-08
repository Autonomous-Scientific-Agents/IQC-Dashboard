# Hammett ortho position bug (position 6 mislabeled as para)

## Location
`descriptor_kit/core/hammett.py:133`

## The bug

```python
# descriptor_kit/core/hammett.py (around line 133)
if position in (4, 6):
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_p")
```

The function returns a **para** Hammett constant (`sigma_p`) whenever the
position is 4 **or 6**. Position 6 is *ortho*, not para, so every ortho
substituent silently receives the wrong electronic parameter.

## Why it happens

The rest of the code base agrees that position 6 on the bpy ring is ortho:

- `descriptor_kit/descriptors/reactant.py` (`_POS_CLASS`):
  ```python
  _POS_CLASS = {3: "meta", 4: "para", 5: "meta", 6: "ortho"}
  ```
- The module docstring at `descriptor_kit/core/hammett.py:138` explicitly
  says: `"6 (ortho, skipped per spec)"`.

So the intent is clearly:
- 3, 5 -> meta
- 4    -> para
- 6    -> ortho (no tabulated sigma_o -> skip / NaN)

But the runtime code lumps 4 and 6 together and hands both the `sigma_p`
value from the table. Every downstream sum/average that includes an ortho
substituent is therefore silently wrong.

## Blast radius

Any descriptor that consumes `sigma_for_fragment` for a position-6
substituent is affected. Concretely, this includes at least:

- `reac_sigma_ortho_pyA` / `reac_sigma_ortho_pyB`
- Any D1/D3 (delta-sum) descriptor that folds ortho into its total
- Downstream ML features / correlations that use these columns

The failure is silent: no exception, no NaN, no warning. The numbers just
come out plausible but wrong.

## Proposed fix

Two options; option A is safer and matches the module docstring.

### Option A - skip position 6 (recommended, matches spec)

```python
# descriptor_kit/core/hammett.py:133
if position == 4:
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_p")
if position in (3, 5):
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_m")
if position == 6:
    # ortho: no sigma_o column in the current table; return NaN so
    # downstream consumers can decide how to treat it.
    return float("nan")
return None
```

### Option B - route position 6 through a real sigma_o table

If we ever curate a sigma_o table:

```python
if position == 4:
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_p")
if position in (3, 5):
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_m")
if position == 6:
    return _SIGMA_TABLE.get(smiles, {}).get("sigma_o")  # requires table update
```

Prefer A until sigma_o values have been sourced and reviewed.

## Verification checklist

After patching:

1. Add a regression test that constructs a bpy substrate with a known
   ortho substituent (e.g. 6-methyl-2,2'-bipyridine) and asserts:
   - `sigma_for_fragment(..., position=6)` is NaN (option A) or matches the
     tabulated sigma_o (option B), NOT the sigma_p value.
2. Recompute a small slice of the precomputed descriptor parquet and diff
   `reac_sigma_ortho_pyA/B` against the previous run - non-zero rows are
   exactly the ones that were wrong.
3. Grep for other callers that pass `position in (4, 6)` or assume the old
   behavior; none should exist, but confirm.

## Notes

- The bug has been present since the current `_POS_CLASS` mapping was
  introduced; any historical precomputed parquet with ortho substituents
  should be regenerated after the fix lands.
- Consider adding an assertion in `_perpos_sigma` that
  `_POS_CLASS[position]` and the branch taken in `sigma_for_fragment` agree,
  so a future regression is caught immediately.
