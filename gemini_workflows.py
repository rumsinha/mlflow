import os
import mlflow
from mlflow.entities import SpanType
from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY) from a .env file if available
load_dotenv()

# Verify that the Gemini API Key is available
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set.")
    print("Please set the GEMINI_API_KEY environment variable or create a .env file in the workspace.")

# =====================================================================
# 1. CONFIGURE MLFLOW TRACKING & AUTOLOGGING
# =====================================================================
# Tell MLflow where the tracking server is located (our local server on port 5001)
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Gemini MLflow Workflows")

# Enable autologging for both Gemini (google-genai SDK) and LangChain
mlflow.gemini.autolog()
mlflow.langchain.autolog()

# Import the Google GenAI SDK
from google import genai
from google.genai import types

# Initialize the Gemini client
client = genai.Client()

# Import LangChain integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# =====================================================================
# WORKFLOW 1: Direct Gemini API Calls with Nested Spans
# =====================================================================
@mlflow.trace(span_type=SpanType.CHAIN)
def run_gemini_workflow(question: str):
    """
    Executes a simple Q&A workflow using the google-genai SDK.
    Uses @mlflow.trace decorators to log custom child spans.
    """
    print(f"\n🚀 Running Workflow 1 (Direct API) with query: '{question}'")
    
    # Generate content structure
    contents = build_messages(question)
    
    # MLflow automatically generates a span for the Gemini invocation under the hood
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            system_instruction="You are a helpful chatbot."
        )
    )
    return parse_response(response)

@mlflow.trace
def build_messages(question: str):
    """
    Preprocesses the user input for the Gemini model.
    Traced as a helper child span.
    """
    return question

@mlflow.trace
def parse_response(response):
    """
    Extracts the text content from the Gemini model response.
    Traced as a helper child span.
    """
    return response.text

# =====================================================================
# WORKFLOW 2: Multi-Provider/Multi-Step Workflow using Gemini & LangChain
# =====================================================================
@mlflow.trace(span_type=SpanType.CHAIN)
def gemini_multi_provider_workflow(query: str):
    """
    A multi-step workflow showcasing tool/LLM coordination.
    1. First call: Uses the google-genai SDK directly to extract key topics.
    2. Second call: Uses LangChain's ChatGoogleGenerativeAI to construct a detailed response.
    """
    print(f"\n🚀 Running Workflow 2 (Multi-Step Gemini + LangChain) with query: '{query}'")
    
    # Step 1: Use Gemini directly for initial topic extraction / analysis
    print("[Step 1] Requesting topic extraction via google-genai SDK...")
    analysis = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction="Analyze the query and extract key topics as a comma-separated list."
        )
    )
    topics = analysis.text
    print(f"👉 Extracted Topics: {topics.strip()}")

    # Step 2: Use LangChain with Gemini for structured response generation
    print("[Step 2] Processing final prompt via LangChain + ChatGoogleGenerativeAI...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = ChatPromptTemplate.from_template(
        "Based on these topics: {topics}\nGenerate a detailed response to: {query}"
    )
    
    # Combine prompt and LLM into a chain
    chain = prompt | llm
    
    # Invoke the chain; this step will generate a LangChain span containing LLM metadata
    response = chain.invoke({"topics": topics, "query": query})

    return response.content

# =====================================================================
# EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    if not api_key:
        print("\n❌ Cannot execute live API calls without GEMINI_API_KEY.")
        print("Please export GEMINI_API_KEY='your_key' and try again.")
        exit(1)
        
    try:
        # Run Workflow 1
        w1_result = run_gemini_workflow("What is MLflow?")
        print("\n=== Workflow 1 Response ===")
        print(w1_result)
        
        # Run Workflow 2
        w2_result = gemini_multi_provider_workflow("Explain quantum computing")
        print("\n=== Workflow 2 Response ===")
        print(w2_result)
        
        print("\n🎉 Both workflows completed successfully!")
        print("Traces have been successfully logged to the MLflow server.")
        print("View them in the MLflow UI at: http://127.0.0.1:5001")
        
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
