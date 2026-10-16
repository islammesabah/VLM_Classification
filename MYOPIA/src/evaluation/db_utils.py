"""
Shared de-duplication for the per-dataset answer databases.

Partway through the study the GTSRB image directory was reorganised from
`Data/GTSRB/all_sequences_training/` to `datasets/GTSRB/Training/`. Thirty
image-model pairs in the primary condition (126 rows once every temperature
and prompt configuration is counted) were re-evaluated under the new layout
without the superseded rows being cleared, so the same physical image appears
twice for the same model and configuration under two different path strings.

Counting rows directly therefore double-counts those images: the GTSRB
instance totals came to 2,733 rather than 901 images x 3 models = 2,703.
The duplicated rows are genuine independent re-runs at temperature 0.7, so
the two copies do not always agree, and the choice of which to keep matters.

We keep the row under the current `datasets/GTSRB/Training/` layout, since
that is the path the released code resolves, and drop its legacy-prefixed
twin. Every one of the 126 duplicate groups contains exactly one row of each
kind, so the rule is unambiguous.

Use `ANSWERS` anywhere a query would otherwise read `FROM answers`:

    cur.execute(f"SELECT Class, LLM_name FROM {ANSWERS} WHERE ...")

The expression is a no-op on the MNIST, CIFAR-10 and MSCOCO databases, which
contain no path variants, so it is safe to apply uniformly.
"""

# Trailing "NNNNN/NNNNN_NNNNN.jpeg" uniquely identifies a GTSRB image and is
# identical across both directory layouts, so the twin is matched on it
# rather than on a fixed prefix length.
_IMAGE_KEY_CHARS = 22

LEGACY_PREFIX = "Data/GTSRB/all_sequences_training/"
CANONICAL_PREFIX = "datasets/GTSRB/Training/"

ANSWERS = f"""(
    SELECT * FROM answers a
    WHERE a.Image_path NOT LIKE '{LEGACY_PREFIX}%'
       OR NOT EXISTS (
            SELECT 1 FROM answers b
            WHERE b.Image_path LIKE '{CANONICAL_PREFIX}%'
              AND substr(b.Image_path, -{_IMAGE_KEY_CHARS}) = substr(a.Image_path, -{_IMAGE_KEY_CHARS})
              AND b.LLM_name = a.LLM_name
              AND b.temperature = a.temperature
              AND b.Include_memory = a.Include_memory
              AND b.Include_description = a.Include_description
              AND b.RunId = a.RunId
       )
)"""
