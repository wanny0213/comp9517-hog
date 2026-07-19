"""
iNaturalist-2021-mini subset builder.

Randomly selects N species from the dataset, extracts ONLY those species' images
from the tar.gz archives (no full unpack), and writes:

    dataset/
        train/  <species_name>/<img>.jpg   (40 imgs/species from train_mini)
        val/    <species_name>/<img>.jpg   (10 imgs/species from train_mini)
        test/   <species_name>/<img>.jpg   (≥10 imgs/species from val.tar.gz)

Also writes dataset/manifest.json with species list, seed, and per-class counts.

Usage:
    python build_subset.py \\
        --train_json  /path/to/train_mini.json \\
        --val_json    /path/to/val.json \\
        --train_tar   /path/to/train_mini.tar.gz \\
        --val_tar     /path/to/val.tar.gz \\
        --out_dir     dataset \\
        --n_species   500 \\
        --seed        42

Notes:
  - The annotation files downloaded from the iNat2021 repo are themselves
    tar.gz archives (train_mini.json.tar.gz / val.json.tar.gz).  Extract them
    first:  tar -xzf train_mini.json.tar.gz  →  train_mini.json
  - Fixed seed guarantees every team member gets the identical subset.
  - Selective tar extraction: only members matching selected species are read,
    so you never need to fully decompress 42 GB.
  - Species with fewer than MIN_TRAIN_IMGS (50) or MIN_TEST_IMGS (10) are
    silently skipped and a replacement is drawn.
"""
import argparse
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path


MIN_TRAIN_IMGS = 50   # must have this many in train_mini to be eligible
MIN_TEST_IMGS  = 10   # must have this many in val to be included
N_TRAIN        = 40   # images written to dataset/train/
N_VAL          = 10   # images written to dataset/val/  (total train+val = 50)


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_json(json_path: str) -> tuple[dict, dict, dict]:
    """Return (img_id→file_name, img_id→cat_id, cat_id→cat_name)."""
    print(f"Parsing {json_path} …")
    with open(json_path) as f:
        data = json.load(f)

    img_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    img_to_cat  = {ann["image_id"]: ann["category_id"] for ann in data["annotations"]}
    cat_to_name = {cat["id"]: cat["name"].replace(" ", "_") for cat in data["categories"]}
    return img_to_file, img_to_cat, cat_to_name


def group_by_category(img_to_file: dict, img_to_cat: dict) -> dict[int, list]:
    """Map category_id → list of file_name strings."""
    cat_files: dict[int, list] = defaultdict(list)
    for img_id, cat_id in img_to_cat.items():
        fname = img_to_file.get(img_id)
        if fname:
            cat_files[cat_id].append(fname)
    return cat_files


# ── species selection ─────────────────────────────────────────────────────────

def select_species(
    train_cat_files: dict[int, list],
    test_cat_files:  dict[int, list],
    cat_to_name: dict[int, str],
    n_species: int,
    seed: int,
) -> list[int]:
    """Return a sorted list of category_ids meeting both train and test thresholds."""
    rng = random.Random(seed)
    eligible = [
        cat_id for cat_id in train_cat_files
        if len(train_cat_files[cat_id]) >= MIN_TRAIN_IMGS
        and len(test_cat_files.get(cat_id, [])) >= MIN_TEST_IMGS
    ]
    if len(eligible) < n_species:
        raise ValueError(
            f"Only {len(eligible)} species meet the eligibility criteria "
            f"(need ≥{MIN_TRAIN_IMGS} train, ≥{MIN_TEST_IMGS} test). "
            f"Requested {n_species}."
        )
    selected = sorted(rng.sample(eligible, n_species))
    print(f"Selected {len(selected)} species (seed={seed}).")
    return selected


# ── selective tar extraction ──────────────────────────────────────────────────

def extract_files(tar_path: str, file_to_dest: dict[str, Path]):
    """Single-pass scan of tar_path; extract only members present in file_to_dest.

    Keys of file_to_dest are normalised tar member names (leading "./" stripped).
    Does not require a pre-built index — one sequential read is enough for gz archives.
    """
    print(f"Extracting {len(file_to_dest)} files from {tar_path} …")
    remaining = set(file_to_dest.keys())
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not remaining:
                break
            name = member.name.lstrip("./")
            if name not in remaining:
                continue
            dest = file_to_dest[name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            fobj = tf.extractfile(member)
            if fobj is None:
                remaining.discard(name)
                continue
            dest.write_bytes(fobj.read())
            remaining.discard(name)
            extracted += 1
            if extracted % 1000 == 0:
                print(f"  … {extracted}/{len(file_to_dest)}", flush=True)
    if remaining:
        print(f"  [warn] {len(remaining)} files not found in archive.")
    print(f"  Extracted {extracted} files.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build iNaturalist species subset")
    parser.add_argument("--train_json", required=True, help="Path to train_mini.json")
    parser.add_argument("--val_json",   required=True, help="Path to val.json")
    parser.add_argument("--train_tar",  required=True, help="Path to train_mini.tar.gz")
    parser.add_argument("--val_tar",    required=True, help="Path to val.tar.gz")
    parser.add_argument("--out_dir",    default="dataset", help="Output root directory")
    parser.add_argument("--n_species",  type=int, default=500, help="Number of species to select")
    parser.add_argument("--seed",       type=int, default=42,  help="Random seed (share with team!)")
    parser.add_argument("--dry_run",    action="store_true", help="Print plan without extracting")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 1. Parse JSONs
    train_img_to_file, train_img_to_cat, cat_to_name = parse_json(args.train_json)
    val_img_to_file,   val_img_to_cat,   _           = parse_json(args.val_json)

    train_cat_files = group_by_category(train_img_to_file, train_img_to_cat)
    test_cat_files  = group_by_category(val_img_to_file,   val_img_to_cat)

    # 2. Select species
    selected_cats = select_species(
        train_cat_files, test_cat_files, cat_to_name,
        args.n_species, args.seed
    )

    # 3. Assign images to splits
    rng = random.Random(args.seed)
    train_file_to_dest: dict[str, Path] = {}
    test_file_to_dest:  dict[str, Path] = {}
    manifest_species = []

    for cat_id in selected_cats:
        species_name = cat_to_name[cat_id]
        safe_name = species_name.replace("/", "_")

        # train+val from train_mini: shuffle deterministically then split 40/10
        imgs = list(train_cat_files[cat_id])
        rng_local = random.Random(args.seed ^ cat_id)
        rng_local.shuffle(imgs)
        train_imgs = imgs[:N_TRAIN]
        val_imgs   = imgs[N_TRAIN: N_TRAIN + N_VAL]

        for i, fname in enumerate(train_imgs):
            norm = fname.lstrip("./")
            dest = out_dir / "train" / safe_name / f"{i:04d}.jpg"
            train_file_to_dest[norm] = dest

        for i, fname in enumerate(val_imgs):
            norm = fname.lstrip("./")
            dest = out_dir / "val" / safe_name / f"{i:04d}.jpg"
            train_file_to_dest[norm] = dest  # still from train_mini.tar.gz

        # test from val.tar.gz
        test_imgs = list(test_cat_files[cat_id])
        rng_local.shuffle(test_imgs)
        test_imgs = test_imgs[:max(MIN_TEST_IMGS, len(test_imgs))]  # take all available up to quota
        for i, fname in enumerate(test_imgs):
            norm = fname.lstrip("./")
            dest = out_dir / "test" / safe_name / f"{i:04d}.jpg"
            test_file_to_dest[norm] = dest

        manifest_species.append({
            "category_id": cat_id,
            "name": species_name,
            "n_train": len(train_imgs),
            "n_val": len(val_imgs),
            "n_test": len(test_imgs),
        })

    print(f"\nPlan: {len(train_file_to_dest)} train/val extractions, "
          f"{len(test_file_to_dest)} test extractions.")

    if args.dry_run:
        print("Dry run — no files written.")
        print("First 5 train destinations:")
        for k, v in list(train_file_to_dest.items())[:5]:
            print(f"  {k}  →  {v}")
        return

    # 4. Extract
    extract_files(args.train_tar, train_file_to_dest)
    extract_files(args.val_tar,   test_file_to_dest)

    # 5. Write manifest
    manifest = {
        "seed": args.seed,
        "n_species": args.n_species,
        "n_train_per_class": N_TRAIN,
        "n_val_per_class": N_VAL,
        "species": manifest_species,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Dataset written to {out_dir}/")
    print(f"Manifest: {manifest_path}")
    print(f"Species count: {len(manifest_species)}")
    print(f"Share manifest.json + seed={args.seed} with your team for reproducibility.")


if __name__ == "__main__":
    main()
