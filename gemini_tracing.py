import os
import mlflow
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Specify the tracking URI for the MLflow server.
mlflow.set_tracking_uri("http://localhost:5001")

# Specify the experiment you just created for your LLM application or AI agent.
mlflow.set_experiment("My App")

# Enable automatic tracing for all Gemini API calls.
mlflow.gemini.autolog()

# Initialize the Gemini client.
# It automatically picks up the GEMINI_API_KEY environment variable.
client = genai.Client()

# The trace of the following is sent to the MLflow server.
print("Sending request to Gemini model...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather like in Seattle?",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful weather assistant."
    )
)

print("\nResponse from Gemini:")
print(response.text)
