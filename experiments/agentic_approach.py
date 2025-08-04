from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from PIL import Image
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen import UserProxyAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination



# Define a tool that searches the web for information.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


# Create an agent that uses the Groq llama3-70b-8192 model.
custom_model_client = OpenAIChatCompletionClient(
    model="llama3-70b-8192",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
)
vison_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    model_info={
        "vision": True,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "player",
    model_client=custom_model_client,
    system_message="""You are a plyer agent who paly a Yes/No 20 Question Game with the vision agent. Vision Agent has an Image of street sign and you will ask the agent a visual simple questions for classifying street signs. 
    The vision agent could make mistakes in answering your question so take this into account and continue to ask simple questions about shape, colour patterns, symbols, ..etc to make it easy for vision agent.
    You are also authorized to terminate the game with an "APPROVE" status once the final class has been reached
    you need to know the class of the Image with vision agents from the following classes: 
    {
    "0": "red and white circle 20 kph speed limit",
    "1": "red and white circle 30 kph speed limit",
    "2": "red and white circle 50 kph speed limit",
    "3": "red and white circle 60 kph speed limit",
    "4": "red and white circle 70 kph speed limit",
    "5": "red and white circle 80 kph speed limit",
    "6": "end / de-restriction of 80 kph speed limit",
    "7": "red and white circle 100 kph speed limit",
    "8": "red and white circle 120 kph speed limit",
    "9": "red and white circle red car and black car no passing",
    "10": "red and white circle red truck and black car no passing",
    "11": "red and white triangle road intersection warning",
    "12": "white and yellow diamond priority road",
    "13": "red and white upside down triangle yield right-of-way",
    "14": "stop",
    "15": "empty red and white circle",
    "16": "red and white circle no truck entry",
    "17": "red circle with white horizonal stripe no entry",
    "18": "red and white triangle with exclamation mark warning",
    "19": "red and white triangle with black left curve approaching warning",
    "20": "red and white triangle with black right curve approaching warning",
    "21": "red and white triangle with black double curve approaching warning",
    "22": "red and white triangle rough / bumpy road warning",
    "23": "red and white triangle car skidding / slipping warning",
    "24": "red and white triangle with merging / narrow lanes warning",
    "25": "red and white triangle with person digging / construction / road work warning",
    "26": "red and white triangle with traffic light approaching warning",
    "27": "red and white triangle with person walking warning",
    "28": "red and white triangle with child and person walking warning",
    "29": "red and white triangle with bicyle warning",
    "30": "red and white triangle with snowflake / ice warning",
    "31": "red and white triangle with deer warning",
    "32": "white circle with gray strike bar no speed limit",
    "33": "blue circle with white right turn arrow mandatory",
    "34": "blue circle with white left turn arrow mandatory",
    "35": "blue circle with white forward arrow mandatory",
    "36": "blue circle with white forward or right turn arrow mandatory",
    "37": "blue circle with white forward or left turn arrow mandatory",
    "38": "blue circle with white keep right arrow mandatory",
    "39": "blue circle with white keep left arrow mandatory",
    "40": "blue circle with white arrows indicating a traffic circle",
    "41": "white circle with gray strike bar indicating no passing for cars has ended",
    "42": "white circle with gray strike bar indicating no passing for trucks has ended"
}
    """,
)

# create the vision agent
vision_agent = AssistantAgent(
    "vision",
    model_client=vison_model_client,
    system_message="You are a visual agent that could take a question and image and answer the question on the image content.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")
# Define a termination condition that stops the task if the critic approves.
# termination = MaxMessageTermination(max_messages=20)

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat(
     participants=[primary_agent, vision_agent], termination_condition=text_termination
)

image_path = "Results/9455.png"
img = Image.open(image_path)
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image as AGImage

img = AGImage(img)
multi_modal_message = MultiModalMessage(content=["Get the class of the following image", img], source="user")

async def assistant_run() -> None:
    await Console(
        team.run_stream(
            task=[multi_modal_message]
        )
    )


# Use asyncio.run(assistant_run()) when running in a script.
import asyncio
asyncio.run(assistant_run())
