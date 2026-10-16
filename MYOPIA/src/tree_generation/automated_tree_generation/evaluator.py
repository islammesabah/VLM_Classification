import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prompt_library
from typing import Dict, List, Any, Literal
from pydantic import BaseModel, ValidationError, Field
from openai import OpenAI
from dotenv import load_dotenv
from generator import QuestionEntry
from typing import Optional

from monitor import OpenRouterMonitor
# Load environment variables from the .env file
load_dotenv()

class ScoreEntry(BaseModel):
    score_name: str = Field(
        ...,
        description="The name of the score",
        example="Clarity"
    )
    score_value: int = Field(
        ...,
        description="The score value from 0 to 10",
        example=8
    )

class QuestionEvaluationEntry(BaseModel):
    question_entry: QuestionEntry = Field(
        ...,
        description="The question to be evaluated",
        example={
                "question": "Does the object's outline predominantly consist of straight lines?",
                "explanation": "This question helps to distinguish between classes with straight-lined objects and those with curved outlines.",
                "answers": [{"Yes": [0, 1, 8, 9]},{"No": [2, 3, 4, 5, 6, 7]}]
            }
    )
    scores: List[ScoreEntry] = Field(
        ...,
        description="List of scores for the question based on evaluation criteria"
    )

class EvaluatorOutput(BaseModel):
    question_evaluations: List[QuestionEvaluationEntry] = Field(
        ...,
        description="List of evaluated questions with scores",
        example=[
            {
                "question_entry": {
                    "question": "Does the object's outline predominantly consist of straight lines?",
                    "explanation": "This question helps to distinguish between classes with straight-lined objects and those with curved outlines.",
                    "answers": [{"Yes": [0, 1, 8, 9]},{"No": [2, 3, 4, 5, 6, 7]}]
            },
                "scores": [
                    {"score_name": "Clarity", "score_value": 2},
                    {"score_name": "Completeness", "score_value": 1},
                    {"score_name": "Visual relevance", "score_value": 3},
                    {"score_name": "Creativity and Logical Grouping", "score_value": 1},
                    {"score_name": "Maximizing splits", "score_value": 2},
                    {"score_name": "Overall Score", "score_value": 9}
                ]
            }
        ]
    )
    top_k_questions: List[QuestionEvaluationEntry] = Field(
        ...,
        description="List of top K questions after evaluation",
        example=[
            {
                "question_entry": {
                    "question": "Does the object's outline predominantly consist of straight lines?",
                    "explanation": "This question helps to distinguish between classes with straight-lined objects and those with curved outlines.",
                    "answers": [{"Yes": [0, 1, 8, 9]},{"No": [2, 3, 4, 5, 6, 7]}]
            },
                "scores": [
                    {"score_name": "Clarity", "score_value": 2},
                    {"score_name": "Completeness", "score_value": 1},
                    {"score_name": "Visual relevance", "score_value": 3},
                    {"score_name": "Creativity and Logical Grouping", "score_value": 1},
                    {"score_name": "Maximizing splits", "score_value": 2},
                    {"score_name": "Overall Score", "score_value": 9}
                ]
            }
        ]
    )
    explanation: str = Field(
        ...,
        description="Explanation of the evaluation process and criteria used",
        example="The questions were evaluated based on clarity, completeness, visual relevance, creativity, and maximizing splits among classes. Question3 is highly clear and visually relevant, while Question4 leverages shape for logical grouping. The top K questions were selected based on their overall scores, ensuring they effectively distinguish between classes."
    )

def get_client(provider: Literal["openai", "openrouter"] = "openai") -> OpenAI:
    """
    Initialize and return OpenAI client based on provider choice
    
    Args:
        provider: Either "openai" or "openrouter"
    
    Returns:
        Configured OpenAI client
    """
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return OpenAI(api_key=api_key)
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        return OpenAI(
            api_key=api_key,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'openrouter'")

def get_model_name(provider: Literal["openai", "openrouter"] = "openai") -> str:
    """
    Get the appropriate model name based on provider
    
    Args:
        provider: Either "openai" or "openrouter"
    
    Returns:
        Model name string
    """
    base_model_openai = "openai/gpt-4.1"
    
    if provider == "openai":
        return base_model_openai
    elif provider == "openrouter":
        return f"openai/gpt-4.1"
    else:
        raise ValueError(f"Unknown provider: {provider}")

def current_classes_names(current_classes: List[int], class_names: Dict[str, str]):
    return [(i,class_names[str(i)]) for i in current_classes if str(i) in class_names]

def evaluate_structured_questions(
    current_classes: List[int],
    past_questions: List[str],
    generator_output: str,
    dataset_path: str,
    keep_top_k: int = 2,
    provider: Literal["openai", "openrouter"] = "openai",
    monitor: Optional[OpenRouterMonitor] = None
) -> EvaluatorOutput:
    """
    Evaluates structured questions using OpenAI API or OpenRouter
    
    Args:
        current_classes: List of class indices
        past_questions: List of previously asked questions
        generator_output: JSON string of generated questions
        dataset_path: Path to the dataset containing class_names.json
        keep_top_k: Number of top questions to keep
        provider: API provider to use ("openai" or "openrouter")
    
    Returns:
        EvaluatorOutput with evaluated questions and scores
    """
    # Initialize client based on provider
    client = get_client(provider)
    model_name = get_model_name(provider)
    
    # load class names from dataset-specific file
    try:
        print(f"Loading class names from: {os.path.join(dataset_path, 'class_names.json')}")
        with open(os.path.join(dataset_path, 'class_names.json'), 'r') as file:
            class_names = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading class names: {str(e)}")
    
    # Load and format system prompt with current context
    try:
        system_content = load_system_prompt(
            "prompts_evaluator.json",
            current_classes=current_classes_names(current_classes, class_names),
            class_names=class_names,
            past_questions=past_questions,
            generator_output=generator_output,
            keep_top_k=keep_top_k
        )
    except RuntimeError as e:
        raise ValueError(f"Prompt configuration error: {str(e)}")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Evaluate the questions."}
    ]

    # Add extra headers for OpenRouter (optional but recommended)
    extra_headers = {}
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": os.environ.get("APP_URL", ""),
            "X-Title": os.environ.get("APP_NAME", "Question Evaluator")
        }

    # print(f"Messages Prompt Evaluator: {messages}")
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=EvaluatorOutput,
        **({"extra_headers": extra_headers} if extra_headers else {})
    )

    # TRACK USAGE IF MONITOR PROVIDED
    if monitor and hasattr(response, 'usage'):
        usage = response.usage
        monitor.track_call(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
    
    response_json = response.choices[0].message.content
    print(f"Response JSON: {response_json}")

    try:
        return EvaluatorOutput.model_validate_json(response_json)
    except ValidationError as e:
        print(f"Validation Error: {e.errors()}")
        raise

def load_system_prompt(prompt_file: str, class_names: dict, **format_args) -> str:
    """Load and format a system prompt from the shared prompts/ directory."""
    try:
        format_args['whole_classes'] = class_names
        prompts = prompt_library.load_json(prompt_file)
        template = prompts.get("evaluator_prompt_5_metrics", "")
        return template.format(**format_args)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error loading system prompt: {str(e)}")


if __name__ == "__main__":
    current_classes = [0, 1, 8, 9]
    past_questions = {"Question1": "Does the object's outline predominantly consist of straight lines?", "Answers1": {"Yes": [0, 1, 8, 9],"No": [2, 3, 4, 5, 6, 7]}}
    # print(EvaluatorOutput.model_json_schema())
    past_questions = [f"{k}: {v}" for k, v in past_questions.items()]
    keep_top_k = 2
    dataset_path = "path/to/dataset"  # Update this path
    generator_output = {
        "Question1": "Does the object have wings?",
        "Answers1": {
            "Yes": [0],
            "No": [1, 8, 9]
        },
        "Question2": "Does the object have a hull designed for water travel?",
        "Answers2": {
            "Yes": [8],
            "No": [0, 1, 9]
        },
        "Question3": "Is the object a land vehicle with four or more wheels?",
        "Answers3": {
            "Yes": [1, 9],
            "No": [0, 8]
        },
        "Question4": "Does the object have an elongated body significantly longer than its width?",
        "Answers4": {
            "Yes": [0, 8],
            "No": [1, 9]
        },
        "Question5": "Does the object have a flatbed or open cargo area?",
        "Answers5": {
            "Yes": [9],
            "No": [0, 1, 8]
        }
    }

    generator_output = json.dumps(generator_output, indent=2)
    
    # Choose provider: "openai" or "openrouter"
    provider = "openrouter"
    
    try:
        output = evaluate_structured_questions(
            current_classes=current_classes,
            past_questions=past_questions,
            generator_output=generator_output,
            dataset_path=dataset_path,
            keep_top_k=keep_top_k,
            provider=provider
        )
        print(f"Using provider: {provider}")
        print(output.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")