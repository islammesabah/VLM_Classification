import argparse
import json
import time
from typing import Optional

from generator import generate_structured_questions, AnswerEntry, QuestionEntry
from evaluator import evaluate_structured_questions
from monitor import OpenRouterMonitor


class Node:
    def __init__(self, question=None, answers=None, history=None, current_classes=None):
        # questions is like: "Does the object's outline predominantly consist of straight lines?"
        # answers is like: ["Yes", "No"]
        # history is a list of previous questions +answers asked
        # current_classes is a list of class indices that are still in the game like [0, 1, 8, 9]
        # children is dictionary of answer as key and K Nodes as value e.g. {"Yes": [ChileNode(...), ..], "No": [ChildNode(...), ..]}
        self.question: str = question
        self.answers: list[str] = answers if answers is not None else []
        self.history: list[str] = history if history is not None else []
        self.current_classes = current_classes if current_classes is not None else []
        self.children = {}

    def __repr__(self):
        return f"Node(question={self.question}, answers={self.answers}, history={self.history}, current_classes={self.current_classes}, children={self.children})"


class Tree:
    def __init__(self, root=None):
        self.root: list[Node] = root if root is not None else []


def generate_tree(
    parent_node: Node,
    current_classes: list,
    past_questions: list,
    n_questions_to_generate: int,
    n_questions_to_keep: int,
    dataset_path: str,
    provider: str = "openrouter",
    monitor: Optional[OpenRouterMonitor] = None,
    max_calls: Optional[int] = None,
    call_counter: Optional[list] = None,
):
    """
    Recursively build a decision tree over `current_classes` by alternating an
    LLM question-generation step (generator.py) with an LLM question-scoring
    step (evaluator.py), keeping the top-`n_questions_to_keep` question(s) at
    each node and recursing into each answer's filtered class subset.

    This is the orchestration previously only present, unextracted, inside
    test_small_tree.ipynb -- pulled out here as a standalone, documented,
    reusable script so the tree-generation pipeline can be run on any new
    label set (any dataset_path with a class_names.json) without a notebook.

    `max_calls`/`call_counter` provide an optional hard stop on total
    generator+evaluator API calls across the whole recursion, since this
    function is a real (small) money cost per call.
    """
    if call_counter is None:
        call_counter = [0]

    print(f"Generating tree for node: {parent_node.question} with answers {parent_node.answers} with classes: {current_classes}")

    if len(current_classes) <= 1:
        return [Node(question="", answers=[], history=parent_node.history, current_classes=current_classes)]

    if max_calls is not None and call_counter[0] >= max_calls:
        print(f"max_calls={max_calls} reached; stopping recursion early with an unresolved leaf for classes {current_classes}.")
        return [Node(question="", answers=[], history=parent_node.history, current_classes=current_classes)]

    generated_questions = generate_structured_questions(
        current_classes=current_classes,
        past_questions=past_questions,
        number_of_questions=n_questions_to_generate,
        dataset_path=dataset_path,
        provider=provider,
        monitor=monitor,
    )
    call_counter[0] += 1
    print(f"Generated {n_questions_to_generate} questions.")
    time.sleep(3)  # avoid rate limiting

    if max_calls is not None and call_counter[0] >= max_calls:
        print(f"max_calls={max_calls} reached after generation; stopping before evaluation.")
        return [Node(question="", answers=[], history=parent_node.history, current_classes=current_classes)]

    evaluation_output = evaluate_structured_questions(
        current_classes=current_classes,
        past_questions=past_questions,
        generator_output=generated_questions,
        keep_top_k=n_questions_to_keep,
        dataset_path=dataset_path,
        provider=provider,
        monitor=monitor,
    )
    call_counter[0] += 1
    print(f"Evaluated {n_questions_to_generate} questions and kept {n_questions_to_keep}.")

    top_questions = evaluation_output.top_k_questions
    if len(top_questions) > n_questions_to_keep:
        top_questions = top_questions[:n_questions_to_keep]

    current_nodes = []
    for top_question_entry in top_questions:
        question_entry: QuestionEntry = top_question_entry.question_entry
        question: str = question_entry.question

        answers: list[AnswerEntry] = question_entry.answers
        answers = {answer.key: answer.value for answer in answers}

        question_answers = json.dumps({"question": question, "answers": answers}, indent=None)

        current_node = Node(
            question=question,
            answers=answers.keys(),
            history=parent_node.history + [question_answers],
            current_classes=current_classes,
        )

        for answer in answers.keys():
            filtered_classes: list[int] = answers[answer]
            if filtered_classes:
                children_nodes = generate_tree(
                    current_node,
                    filtered_classes,
                    past_questions + [question_answers],
                    n_questions_to_generate,
                    n_questions_to_keep,
                    dataset_path=dataset_path,
                    provider=provider,
                    monitor=monitor,
                    max_calls=max_calls,
                    call_counter=call_counter,
                )
                current_node.children[answer] = children_nodes

        current_nodes.append(current_node)
    return current_nodes


def visualize_tree(node: Node, level: int = 0, lines: Optional[list] = None):
    if lines is None:
        lines = []
    prefix = "  " * level
    if node.question:
        lines.append(f"{prefix}[L{level}] Q: {node.question}")
        for ans in node.answers:
            for child in node.children.get(ans, []):
                lines.append(f"{prefix}  -> {ans}:")
                visualize_tree(child, level + 1, lines)
    else:
        lines.append(f"{prefix}[L{level}] Leaf Node: {node.current_classes}")
    return lines


def node_to_dict(node: Node) -> dict:
    return {
        "question": node.question,
        "answers": list(node.answers),
        "current_classes": node.current_classes,
        "children": {ans: [node_to_dict(c) for c in children] for ans, children in node.children.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized LLM-based tree generation for any dataset with a class_names.json.")
    parser.add_argument("--dataset_path", required=True, help="Path to a directory containing class_names.json")
    parser.add_argument("--n_questions_to_generate", type=int, default=3)
    parser.add_argument("--n_questions_to_keep", type=int, default=1)
    parser.add_argument("--provider", default="openrouter", choices=["openai", "openrouter"])
    parser.add_argument("--max_calls", type=int, default=None, help="Hard stop on total generator+evaluator API calls")
    parser.add_argument("--output_prefix", default="generated_tree", help="Prefix for output .json/.txt files")
    args = parser.parse_args()

    with open(f"{args.dataset_path}/class_names.json", "r") as f:
        class_names = json.load(f)
    current_classes = sorted(int(k) for k in class_names.keys())

    monitor = OpenRouterMonitor()
    root_node = Node(question="", answers=[], history=[], current_classes=current_classes)

    call_counter = [0]
    result_nodes = generate_tree(
        parent_node=root_node,
        current_classes=current_classes,
        past_questions=[],
        n_questions_to_generate=args.n_questions_to_generate,
        n_questions_to_keep=args.n_questions_to_keep,
        dataset_path=args.dataset_path,
        provider=args.provider,
        monitor=monitor,
        max_calls=args.max_calls,
        call_counter=call_counter,
    )

    monitor.print_summary()

    lines = []
    for n in result_nodes:
        lines.extend(visualize_tree(n))
    tree_text = "\n".join(lines)
    print(tree_text)

    with open(f"{args.output_prefix}.txt", "w") as f:
        f.write(tree_text)
    with open(f"{args.output_prefix}.json", "w") as f:
        json.dump([node_to_dict(n) for n in result_nodes], f, indent=2)

    print(f"\nTotal generator+evaluator calls: {call_counter[0]}")
    print(f"Saved {args.output_prefix}.txt and {args.output_prefix}.json")
