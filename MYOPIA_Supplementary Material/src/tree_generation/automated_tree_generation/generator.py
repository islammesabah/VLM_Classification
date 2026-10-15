import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import prompt_library
from typing import List, Literal
from pydantic import BaseModel, ValidationError, Field
from openai import OpenAI
from dotenv import load_dotenv
from monitor import OpenRouterMonitor

# Load environment variables from the .env file
load_dotenv()

class AnswerEntry(BaseModel):
    key: str = Field(
        ...,
        description="The answer option for the question",
        example="Yes"
    )
    value: List[int] = Field(
        ...,
        description="List of class indices corresponding to the answer option",
        example=[0, 1, 8, 9]
    )

class QuestionEntry(BaseModel):
    question: str = Field(
        ...,
        description="The question to be answered",
        example="Does the object's outline predominantly consist of straight lines?"
    )
    explanation: str = Field(
        ...,
        description="Explanation of the question's purpose",
        example="This question helps to distinguish between classes with straight-lined objects and those with curved outlines."
    )
    answers: List[AnswerEntry] = Field(
        ...,
        description="Possible answers to the question with corresponding class indices",
        example=[{"Yes": [0, 1, 8, 9]},{"No": [2, 3, 4, 5, 6, 7]}]
    )

class GeneratedQuestionsOutput(BaseModel):
    questions: List[QuestionEntry] = Field(
        ...,
        description="List of generated questions with answers and explanations",
        example=[
            {
                "question": "Does the object's outline predominantly consist of straight lines?",
                "explanation": "This question helps to distinguish between classes with straight-lined objects and those with curved outlines.",
                "answers": [{"Yes": [0, 1, 8, 9]},{"No": [2, 3, 4, 5, 6, 7]}]
            }
        ]
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
        raise ValueError(f"Unknown provider: {provider}, Use either openai or openrouter")

def current_classes_names(current_classes: List[int], class_names: dict):    
    return [(i,class_names[str(i)]) for i in current_classes if str(i) in class_names]

def load_system_prompt(prompt_file: str, **format_args) -> str:
    """Load and format a system prompt from the shared prompts/ directory."""
    try:
        prompts = prompt_library.load_json(prompt_file)
        template = prompts.get("Generator example with evaluator instructions", "")
        return template.format(**format_args)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error loading system prompt: {str(e)}")

def generate_structured_questions(
    current_classes: List[int],
    past_questions: List[str],
    number_of_questions: int,
    dataset_path: str,
    provider: Literal["openai", "openrouter"] = "openai",
    monitor: OpenRouterMonitor = None
) -> GeneratedQuestionsOutput:
    """
    Generates structured questions with dynamic answers using OpenAI API or OpenRouter
    
    Args:
        current_classes: List of class indices to generate questions for
        past_questions: List of previously asked questions
        number_of_questions: Number of questions to generate
        dataset_path: Path to the dataset containing class_names.json
        provider: API provider to use ("openai" or "openrouter")
    
    Returns:
        GeneratedQuestionsOutput with generated questions
    """
    # Initialize client based on provider
    client = get_client(provider)
    model_name = get_model_name(provider)
    
    # load the dataset class names
    try:
        with open(f'{dataset_path}/class_names.json', 'r') as file:
            class_names = json.load(file)
    except FileNotFoundError:
        raise ValueError(f"Class names file not found for dataset: {dataset_path}")
    
    # Load and format system prompt with current context
    try:
        system_content = load_system_prompt(
            "prompts_generator.json",
            num_questions=number_of_questions,
            current_classes=current_classes_names(current_classes, class_names),
            past_questions=past_questions
        )
    except RuntimeError as e:
        raise ValueError(f"Prompt configuration error: {str(e)}")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Generate the requested questions."}
    ]
    
    # Add extra headers for OpenRouter (optional but recommended)
    extra_headers = {}
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": os.environ.get("APP_URL", ""),
            "X-Title": os.environ.get("APP_NAME", "Question Generator")
        }
    
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=GeneratedQuestionsOutput,
        **({"extra_headers": extra_headers} if extra_headers else {})
    )

    if monitor and hasattr(response, 'usage'):
        usage = response.usage
        monitor.track_call(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
    response_json = response.choices[0].message.content

    try:
        return GeneratedQuestionsOutput.model_validate_json(response_json)
    except ValidationError as e:
        print(f"Validation Error: {e.errors()}")
        raise

if __name__ == "__main__":
    current_classes = [0, 1, 8, 9]
    past_questions = {"Question1": "Does the object's outline predominantly consist of straight lines?", "Answers1": {"Yes": [0, 1, 8, 9],"No": [2, 3, 4, 5, 6, 7]}}
    past_questions = [f"{k}: {v}" for k, v in past_questions.items()]
    number_of_questions = 5
    dataset_path = "datasets/CIFAR-10"
    
    # Choose provider: "openai" or "openrouter"
    provider = "openrouter"

    try:
        output = generate_structured_questions(
            current_classes,
            past_questions,
            number_of_questions,
            dataset_path,
            provider=provider
        )
        print(f"Using provider: {provider}")
        print(output.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")