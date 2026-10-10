# Datasets

No image data is committed to this repository. `data/download.py` downloads,
normalises and verifies everything. After it runs, every dataset uses the same layout:

```
<data_root>/
├── lol_v1/{train,test}/{low,high}/*.png            485 / 15 pairs
├── lol_v2_real/{train,test}/{low,high}/*.png       689 / 100 pairs
├── lol_v2_synthetic/{train,test}/{low,high}/*.png  900 / 100 pairs
└── DICM | LIME | MEF | NPE | VV/test/low/*         no ground truth
```

Low and high images in a pair have the same filename. File-name prefixes such as
`low00001.png` / `normal00001.png` in LOL-v2-Real are stripped to `00001.png`.

## Sources and fallback order

For each dataset the script tries these in order and stops at the first copy that
passes verification:

1. **Drive cache**: `<cache_dir>/<name>.zip`, written by an earlier successful run.
2. **Manual file**: any matching zip you put in `<cache_dir>/manual/` yourself
   (see "If Google Drive refuses the download" below).
3. The **internet sources** in the table, in the order listed.

| Dataset | First choice | Backup(s) |
|---|---|---|
| LOL-v1 | HF `geekyrakshit/LoL-Dataset`, then the LOL-v1 part of HF `apeanut/data4LLIE` | Google Drive file `1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H` (Retinexformer README) |
| LOL-v2 Real / Synthetic | HF `apeanut/data4LLIE` | Google Drive file `1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U` (Retinexformer README) |
| DICM, LIME, MEF, NPE, VV | Google Drive folder `1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp` (Retinexformer README, "datasets that have no ground truth") | Google Drive folder `0B_FjaR958nw_djVQanJqeEhUM1k` (BIMEF README); DICM only: `http://mcl.korea.ac.kr/projects/LDR/LDR_TEST_IMAGES_DICM.zip` |

The Google Drive IDs were checked against the
[Retinexformer README](https://github.com/caiyuanhao1998/Retinexformer) and the
[BIMEF README](https://github.com/PilotWang/BIMEF) on 2026-10-09.
The file layout inside the Hugging Face repos is **not hard-coded**. At run time the
script lists the repo's files with `huggingface_hub.list_repo_files`, downloads only
the matching images and archives with `snapshot_download(allow_patterns=...)`, then
finds the low/high and train/test folders by name:

| Meaning | Accepted folder names (case-insensitive) |
|---|---|
| low-light input | `low`, `input`, `lq`, `dark`, ... |
| ground truth | `high`, `normal`, `gt`, `target`, ... |
| train split | `train`, `our485`, `training`, ... |
| test split | `test`, `eval15`, `eval`, `val`, ... |

## Verification

`download.py` checks each paired dataset for:

- the expected pair count per split (the numbers above),
- every low image having a high image with the same name, and the reverse,
- the low and high images in each pair having identical sizes.

It prints a summary table and writes `results/data/data_summary.json`, plus one
preview grid per dataset to `results/data/preview_<name>.png` (low-light test images
on the top row, ground truth below).

The unpaired sets have no fixed reference counts; different mirrors ship slightly
different versions. Their image counts are reported but not enforced.

## If Google Drive refuses the download

Popular public Drive files often hit a daily download quota
("Too many users have viewed or downloaded this file recently"). The script detects
this and prints instructions:

1. Open the printed Drive link in a browser, signed in to the Google account you use in Colab.
2. Download the file. For a folder, select everything, then right-click and choose *Download*.
3. Upload the zip to `MyDrive/lowlight-attn-unet/datasets/manual/`, using a name such as
   `LOL-v1.zip`, `LOL-v2.zip` or `unpaired.zip`.
4. Re-run the notebook cell. The manual folder is checked before the internet.

## Licenses and citations

None of these datasets ships with a standard open-source license file. They are
distributed by their authors for **non-commercial academic research**, and that is
how this project uses them. Cite the original papers below whenever you use them,
and do not redistribute the images. If you need certainty for another use, check
with the authors.

| Dataset | Content | Citation |
|---|---|---|
| LOL (v1) | 500 real low/normal-light pairs, 600×400 | C. Wei, W. Wang, W. Yang, J. Liu. *Deep Retinex Decomposition for Low-Light Enhancement.* BMVC 2018. |
| LOL-v2 (Real, Synthetic) | 789 real pairs (600×400); 1000 synthetic pairs (384×384) | W. Yang, W. Wang, H. Huang, S. Wang, J. Liu. *Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement.* IEEE TIP 30, 2021. |
| DICM | 69 images from consumer cameras | C. Lee, C. Lee, C.-S. Kim. *Contrast Enhancement Based on Layered Difference Representation of 2D Histograms.* IEEE TIP 22(12), 2013. |
| LIME | 10 low-light images | X. Guo, Y. Li, H. Ling. *LIME: Low-Light Image Enhancement via Illumination Map Estimation.* IEEE TIP 26(2), 2017. |
| MEF | 17 multi-exposure sequences (under-exposed frames used) | K. Ma, K. Zeng, Z. Wang. *Perceptual Quality Assessment for Multi-Exposure Image Fusion.* IEEE TIP 24(11), 2015. |
| NPE | 8 non-uniformly lit images | S. Wang, J. Zheng, H.-M. Hu, B. Li. *Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images.* IEEE TIP 22(9), 2013. |
| VV | 24 challenging back-lit / low-light photos | V. Vonikakis. *Busting image enhancement and tone-mapping algorithms* (dataset web page, https://sites.google.com/site/vonikakis/datasets). |

The image counts above are as described by the authors. The numbers that matter for
this project are those `download.py` measures and records in `results/data/data_summary.json`.

**Overlap note:** LOL-v2-Real is built on top of the LOL-v1 captures, so the two
datasets share images. `download.py` measures this by hashing decoded pixels and
prints how many LOL-v1 train/test images reappear in each LOL-v2-Real split. The
counts are also stored under `lol_v1_v2_overlap` in `data_summary.json`. Never train
on one of these datasets and report test numbers on the other without checking and
stating the overlap.

Mirror acknowledgements: the Retinexformer authors (Cai et al., ICCV 2023) and the
BIMEF authors (Ying et al., 2017) for hosting the Google Drive copies, and the Hugging
Face uploaders `geekyrakshit` and `apeanut`.
