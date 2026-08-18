# Data contract

PriST-RIS has one immutable data interpretation.

## Quasi-static

- Train HDF5 keys: `input_da`, `output_da`.
- Validation/test keys: `Yd`, `Hd`.
- Loader output: observation `[B,1,32,64,2]`, target `[B,1,256,64,2]`.

## Mobility

- Keys: `Yd`, `Hd`.
- Exact counts: train 20,000; validation 1,800; test 9,000.
- Loader output: observation `[B,2,32,64,2]`, target `[B,6,256,64,2]`.
- Query blocks belong to the same sample; no cross-sample sequence is constructed.

The first half of the raw leading channel dimension is real and the second half is imaginary. The loader stacks them into the final complex axis. The 32 observations correspond to RIS indices `range(0,256,8)`. Grid coordinates follow row-major `index=16*row+column`. Observed times are `[0]` for Quasi and `[0,1]` for Mobility; query times are `[0]` and `[0,1,2,3,4,5]` respectively.

Supported discovery layouts include `<root>/<domain>/<stem>/<file>`, `<root>/<file>`, and the legacy `<root>/risce[/risce-0]/<stem>/<file>` trees. Test construction is denied unless `allow_test=True`, which only the freeze-validated evaluation path supplies.
