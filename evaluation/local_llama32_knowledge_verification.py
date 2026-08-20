"""
Local (on-device, MLX) image-grounded knowledge-verification run for the
*actual* LLaMA-3.2-11B-Vision-Instruct model, since its OpenRouter endpoint
is gone (see main.tex Limitations). Runs the identical GTSRB protocol used
for GPT-4o/Llama-4-Scout/Qwen-VL-Max in
evaluation/image_grounded_knowledge_verification.py, but via a local 8-bit
MLX quantized checkpoint instead of an API call.

At ~90s/call this run is many hours long, so this script is written to be
crash-resilient: every completed image's full node-by-node result is
appended to the output CSV immediately (not buffered), and on restart it
skips (class, image_index) pairs already present in that CSV.

Usage:
    python evaluation/local_llama32_knowledge_verification.py \
        --samples_per_class 5 --max_hours 22
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.data_loader import GTSRBLoader
from main import InferenceConfig

MODEL_ID = "mlx-community/Llama-3.2-11B-Vision-Instruct-8bit"
OUT_PATH = Path("results/image_grounded_verification/gtsrb_llama32_11b_local_image_grounded.csv")
FIELDNAMES = ["dataset", "model", "true_class", "image_index", "depth", "question",
              "expected_answer", "predicted_answer", "correct", "latency_s"]


def path_to_class(tree_root, target_class: int):
    path = []
    node = tree_root
    while node.question:
        chosen = None
        for answer, child in node.children.items():
            if target_class in child.get_all_possible_classes():
                chosen = (answer, child)
                break
        if chosen is None:
            break
        answer, child = chosen
        path.append((node.question, answer))
        node = child
    return path


def extract_answer(answers, response_text):
    if not response_text:
        return None
    sorted_answers = sorted(answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), response_text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def sample_images_per_class(image_paths, labels, samples_per_class, seed=42):
    import random
    rng = random.Random(seed)
    by_class = {}
    for path, label in zip(image_paths, labels):
        by_class.setdefault(int(label), []).append(path)
    sampled = {}
    for cls, paths in by_class.items():
        k = min(samples_per_class, len(paths))
        sampled[cls] = rng.sample(sorted(paths), k)
    return sampled


def already_done():
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["true_class"]), row["image_index"]))
    return done


def append_rows(rows):
    write_header = not OUT_PATH.exists()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def answer_missing(node, answer):
    return answer not in node.children


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_class", type=int, default=5)
    parser.add_argument("--max_hours", type=float, default=22.0)
    args = parser.parse_args()

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print("Loading model...", flush=True)
    t0 = time.time()
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    inf_config = InferenceConfig.from_file("config.json", "GTSRB", model="gpt-4o")
    loader = GTSRBLoader()
    _, labels, class_names, image_paths = loader.GTSRB_load_images(
        inf_config.dataset_path, inf_config.images_dir
    )
    tree_root = loader.build_tree()
    labels = [int(l) for l in labels]

    sampled = sample_images_per_class(image_paths, labels, args.samples_per_class)
    done = already_done()
    print(f"Already completed: {len(done)} images. Target: "
          f"{sum(len(v) for v in sampled.values())} images total.", flush=True)

    start_time = time.time()
    deadline = start_time + args.max_hours * 3600
    n_images_done = 0

    for target_class, paths in sampled.items():
        node_path = path_to_class(tree_root, target_class)
        if not node_path:
            continue
        for img_path in paths:
            image_index = Path(img_path).name
            if (target_class, image_index) in done:
                continue
            if time.time() > deadline:
                print(f"max_hours={args.max_hours} reached; stopping.", flush=True)
                return

            rows = []
            node = tree_root
            for depth, (question, expected_answer) in enumerate(node_path):
                prompt_text = f"{question} Choose one of these answers: {node.answers}."
                prompt = apply_chat_template(processor, config, prompt_text, num_images=1)
                t1 = time.time()
                try:
                    response_text = generate(model, processor, prompt, [img_path],
                                              max_tokens=30, verbose=False)
                except Exception as e:
                    print(f"Generation error on {img_path} depth {depth}: {e}", flush=True)
                    response_text = ""
                latency = time.time() - t1

                predicted_answer = extract_answer(node.answers, response_text)
                correct = predicted_answer == expected_answer
                rows.append({
                    "dataset": "GTSRB", "model": "meta-llama/llama-3.2-11b-vision-instruct-8bit-local",
                    "true_class": target_class, "image_index": image_index, "depth": depth,
                    "question": question, "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer, "correct": correct,
                    "latency_s": round(latency, 1),
                })
                if not correct or answer_missing(node, expected_answer):
                    break
                node = node.children[expected_answer]

            append_rows(rows)
            n_images_done += 1
            elapsed = time.time() - start_time
            print(f"[{n_images_done}] class={target_class} image={image_index} "
                  f"nodes={len(rows)} elapsed={elapsed/60:.1f}min", flush=True)

    print("All sampled images completed.", flush=True)


if __name__ == "__main__":
    main()
