"""
Local (on-device, MLX) full zero-shot + tree-based inference for the actual
LLaMA-3.2-11B-Vision-Instruct model, since its OpenRouter endpoint is gone
(see main.tex Limitations). Fills in LLaMA's missing T=0 rows for MNIST and
MSCOCO -- the only two datasets where Table 3 currently omits LLaMA at T=0
entirely, unlike GTSRB/CIFAR-10 which collected it via OpenRouter before the
endpoint died.

Mirrors main.py's VLMInference.zero_shot_inference / tree_inference logic
exactly (same prompt construction, same regex extraction, same DB schema via
the same DatabaseManager), swapping only the OpenAI-compatible API call for a
local mlx_vlm generate() call. Restricted to the primary condition
(Include_memory=False, Include_description=0, Include_zero_shot_label=False,
RunId=0, temperature=0) -- the only condition needed to fill Table 3.

Tagged in the database as a distinct model id
("meta-llama/llama-3.2-11b-vision-instruct-8bit-local"), matching the
convention already used for the GTSRB local cross-model check, so this
local/quantized data is never silently conflated with the earlier
OpenRouter-hosted LLaMA rows under the plain model id.

At ~90s/call (established by the existing GTSRB local run), this is a very
long job -- crash-resilient by construction: every image is written to the
database immediately via the same DatabaseManager.image_exists() dedup check
main.py itself uses, so a restart after any crash simply skips finished rows.

Usage:
    python evaluation/local_llama32_full_inference.py --dataset MNIST
    python evaluation/local_llama32_full_inference.py --dataset MSCOCO
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from datasets.data_loader import MNISTLoader, MSCOCOLoader
from main import InferenceConfig, DatabaseManager

load_dotenv()

MODEL_ID = "mlx-community/Llama-3.2-11B-Vision-Instruct-8bit"
DB_MODEL_NAME = "meta-llama/llama-3.2-11b-vision-instruct-8bit-local"
LOADERS = {"MNIST": MNISTLoader, "MSCOCO": MSCOCOLoader}
MAX_TOKENS = 100


def extract_class_from_response(response, class_names):
    if not response:
        return -1
    match = re.search(r'^\D*(\d+)', response)
    if match:
        class_id = int(match.group(1))
        return class_id if str(class_id) in class_names else -1
    return -1


def extract_tree_answer(answers, response_text):
    if not response_text:
        return None
    sorted_answers = sorted(answers, key=len, reverse=True)
    pattern = "(?i)(" + "|".join(f"\\b{re.escape(a)}\\b" for a in sorted_answers) + ")"
    m = re.search(pattern.lower(), response_text.lower())
    if not m:
        return None
    return next((a for a in sorted_answers if a.lower() == m.group(1).lower()), None)


def run_zero_shot(model, processor, mlx_config, apply_chat_template, generate,
                   image_path, class_names, prompt_template):
    class_ids = sorted(int(k) for k in class_names.keys())
    class_names_str = ", ".join(f"{i}: {class_names[str(i)]}" for i in class_ids)
    prompt_text = prompt_template.replace("{class_names_str}", class_names_str)
    prompt = apply_chat_template(processor, mlx_config, prompt_text, num_images=1)
    response = generate(model, processor, prompt, [str(image_path)],
                         max_tokens=MAX_TOKENS, verbose=False)
    predicted_class = extract_class_from_response(response, class_names)
    return response or "", predicted_class


def run_tree(model, processor, mlx_config, apply_chat_template, generate,
             image_path, tree_root):
    node = tree_root
    tree_path = []
    while node.question:
        prompt_text = f"{node.question} Choose one of these answers: {node.answers}."
        prompt = apply_chat_template(processor, mlx_config, prompt_text, num_images=1)
        response = generate(model, processor, prompt, [str(image_path)],
                             max_tokens=MAX_TOKENS, verbose=False)
        if not response:
            tree_path.append({"question": node.question, "answer": response or ""})
            return json.dumps(tree_path), -1
        answer = extract_tree_answer(node.answers, response)
        if answer is None or answer not in node.children:
            tree_path.append({"question": node.question, "answer": response.strip()})
            return json.dumps(tree_path), -1
        tree_path.append({"question": node.question, "answer": answer})
        node = node.children[answer]
    predicted_class = node.possible_classes[0]
    return json.dumps(tree_path), predicted_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--max_hours", type=float, default=200.0)
    args = parser.parse_args()

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print("Loading local MLX model...", flush=True)
    t0 = time.time()
    model, processor = load(MODEL_ID)
    mlx_config = load_config(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    config = InferenceConfig.from_file("config.json", args.dataset, model="gpt-4o", temperature=0.0)
    loader = LOADERS[args.dataset]()
    images, labels, class_names, additional_info = loader.load_images(config)
    print(f"Loaded {len(images)} images from dataset '{args.dataset}'", flush=True)

    tree_root = loader.build_tree()
    prompt_template = loader.get_zero_shot_prompt(config.to_dict(), 0)

    db = DatabaseManager(config.database_name)

    start_time = time.time()
    deadline = start_time + args.max_hours * 3600
    n_done = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "current.jpg"
        for image_idx in range(len(images)):
            image_path, sequence = loader.extract_image_info(image_idx, additional_info)

            if db.image_exists(image_path, DB_MODEL_NAME, False, 0, False, 0.0, 0):
                continue
            if time.time() > deadline:
                print(f"max_hours={args.max_hours} reached; stopping.", flush=True)
                db.close()
                return

            img = images[image_idx]
            pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(tmp_path, format="JPEG")

            t_img = time.time()
            zs_response, zs_class = run_zero_shot(
                model, processor, mlx_config, apply_chat_template, generate,
                tmp_path, class_names, prompt_template,
            )
            tree_response, tree_class = run_tree(
                model, processor, mlx_config, apply_chat_template, generate,
                tmp_path, tree_root,
            )

            db.insert_result(
                class_label=labels[image_idx], sequence=sequence, image_path=image_path,
                model_name=DB_MODEL_NAME, tree_result=tree_response,
                zero_shot_result=zs_response, tree_class=tree_class, zero_shot_class=zs_class,
                include_memory=False, include_description=0, include_zero_shot_label=False,
                temperature=0.0, RunId=0,
            )

            n_done += 1
            elapsed = time.time() - start_time
            print(f"[{n_done}] {image_path} zs_class={zs_class} tree_class={tree_class} "
                  f"true={labels[image_idx]} this_image={time.time()-t_img:.1f}s "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    db.close()
    print("Inference completed!", flush=True)


if __name__ == "__main__":
    main()
