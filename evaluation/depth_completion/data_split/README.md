# Depth-completion evaluation splits

KITTI-DC and DDAD are scored on a fixed 150-sample subset of their test sets; iBims-1 (100) and
NYUv2 (654) are scored in full and therefore need no file here.

```text
kittidc_dc.txt    150 of 1000
ddad_dc.txt       150 of 3950
```

Each line is one sample, `rgb gt sparse`, relative to the dataset root. Both subsets are an evenly
spaced draw over the sorted sample ids with no random component, so they can be regenerated or
checked against the data:

```bash
python -m evaluation.depth_completion.make_data_split          # verify
python -m evaluation.depth_completion.make_data_split --write  # rebuild
```
