"""
Zero-shot Chain-of-Thought (CoT) baseline: a single VLM call per image, asked
to reason step by step before giving a final class, with no tree structure
and no branching. Isolates whether *any* form of step-by-step reasoning
underperforms flat zero-shot prompting (Section 4.4's finding for our tree),
or whether that's specific to the tree's rigid, non-backtracking branching.

Sampled at a smaller pilot scale (n_images per dataset/model, class-stratified
like evaluation/variance_probe.py) rather than the full ~850-1000/dataset
primary-condition scale, to get a real, cheap read before committing to a
larger run.

Answer extraction looks for an explicit "Final answer: N" marker (the prompt
asks for one); falls back to the last standalone number in the response if
the marker is missing, since CoT responses may mention other numbers (e.g.
"there are 3 classes") before their conclusion -- unlike terse zero-shot
answers, first-number extraction is not reliable here.

Usage:
    python evaluation/cot_baseline.py --dataset GTSRB --model gpt-4o --n_images 30
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from main import InferenceConfig, VLMInference

load_dotenv()

OUT_DIR = Path("results/cot_baseline")
MODELS = ["gpt-4o", "qwen-vl-max", "llama3.2-vision:11b-instruct-fp16"]
DATASETS = ["GTSRB", "MNIST", "CIFAR-10", "MSCOCO"]

COT_PROMPT = (
    "Classify the object in this image. Think step by step: first describe "
    "the visually distinguishing features you observe, then reason about "
    "which of the following classes best matches those features: "
    "{class_names_str}. "
    "End your response with a new line reading exactly: "
    "Final answer: <class ID number>"
)


def select_subsample(labels, n_images, seed=123):
    """
    Guarantees every class is represented at least once (up to n_images
    permitting), rather than relying on a per-class cap that could, for
    datasets with many classes relative to n_images (MSCOCO: 90, GTSRB: 43),
    leave some classes with zero samples purely by shuffle order.
    """
    rng = random.Random(seed)
    by_class = {}
    for idx, cls in enumerate(labels):
        by_class.setdefault(cls, []).append(idx)
    for idx_list in by_class.values():
        rng.shuffle(idx_list)

    classes = list(by_class.keys())
    rng.shuffle(classes)

    selected = []
    # Pass 1: one image per class, guaranteeing full class coverage.
    for cls in classes:
        if len(selected) >= n_images:
            break
        if by_class[cls]:
            selected.append(by_class[cls].pop())

    # Pass 2: fill remaining budget with additional images, round-robin
    # across classes so extra coverage stays balanced rather than piling
    # onto whichever classes happen to come first.
    while len(selected) < n_images and any(by_class.values()):
        for cls in classes:
            if len(selected) >= n_images:
                break
            if by_class[cls]:
                selected.append(by_class[cls].pop())

    return sorted(selected)


def extract_cot_answer(response, class_names):
    """
    Prefer the explicit "Final answer: N" marker the prompt asks for. If the
    model doesn't produce one (e.g. it answers "None" or refuses), fall back
    to the last standalone number in the response -- but only among numbers
    NOT immediately preceded by "step" or "#", since CoT responses routinely
    contain "Step 1/2/3..." headers whose digits are not answers and would
    otherwise be misread as the final class (observed on LLaMA responses that
    end in a non-numeric refusal after a numbered step list).
    """
    if not response:
        return -1
    m = re.search(r"Final answer:\s*(\d+)", response, re.IGNORECASE)
    if m:
        class_id = int(m.group(1))
    else:
        candidates = []
        for match in re.finditer(r"\b(\d+)\b", response):
            preceding = response[max(0, match.start() - 10):match.start()]
            if re.search(r"step\s*[:#]?\s*$", preceding, re.IGNORECASE):
                continue
            candidates.append(match.group(1))
        class_id = int(candidates[-1]) if candidates else None
    if class_id is None:
        return -1
    return class_id if str(class_id) in class_names else -1


def run(dataset_name, model, n_images, max_calls):
    config = InferenceConfig.from_file("config.json", dataset_name, model=model, temperature=0.7)
    engine = VLMInference(config)
    images, labels, class_names, additional_info = engine.dataset_loader.load_images(config)
    labels = [int(l) for l in labels]
    subsample_idx = select_subsample(labels, n_images)

    class_ids = sorted(int(k) for k in class_names.keys())
    class_names_str = ", ".join(f"{i}: {class_names[str(i)]}" for i in class_ids)
    prompt_text = COT_PROMPT.replace("{class_names_str}", class_names_str)

    records = []
    n_calls = 0
    for idx in subsample_idx:
        if n_calls >= max_calls:
            print(f"max_calls={max_calls} reached; stopping.", flush=True)
            break
        image_path, _ = engine.dataset_loader.extract_image_info(idx, additional_info)
        base64_image = engine._encode_image(images[idx])

        payload = {
            "model": engine.model_client.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }],
            "max_tokens": 300,
            "temperature": 0.7,
        }
        try:
            completion = engine.model_client.client.chat.completions.create(**payload)
            response = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"error on {image_path}: {e}", flush=True)
            response = ""
        n_calls += 1

        predicted = extract_cot_answer(response, class_names)
        correct = predicted == labels[idx]
        records.append({
            "dataset": dataset_name, "model": model, "image_path": image_path,
            "true_class": labels[idx], "predicted_class": predicted, "correct": correct,
            "response": response,
        })
        print(f"[{n_calls}/{len(subsample_idx)}] {image_path} pred={predicted} "
              f"true={labels[idx]} correct={correct}", flush=True)
        time.sleep(0.5)

    df = pd.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "_").replace(":", "_").replace("-", "_")
    out_path = OUT_DIR / f"{dataset_name.replace('-', '').lower()}_{model_slug}_cot.csv"
    df.to_csv(out_path, index=False)
    acc = df["correct"].mean() if len(df) else float("nan")
    print(f"\n[{dataset_name}/{model}] n={len(df)} accuracy={acc*100:.1f}% -> {out_path}", flush=True)
    engine.cleanup()
    return acc, len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--n_images", type=int, default=30)
    parser.add_argument("--max_calls", type=int, default=100)
    args = parser.parse_args()
    run(args.dataset, args.model, args.n_images, args.max_calls)


if __name__ == "__main__":
    main()
