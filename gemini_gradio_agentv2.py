import os
from dotenv import load_dotenv

# 1. Load environment variables from .env if present
load_dotenv()

# Read the username to a local variable for path creation, then remove it from environment
# to prevent conflicts with Service Principal auth in the Databricks SDK.
default_user = os.environ.get("DATABRICKS_USERNAME", "Shared")

# 2. Set MLflow environment variable to enable Databricks SDK authentication BEFORE importing mlflow
os.environ["DATABRICKS_AUTH_TYPE"] = "pat"
os.environ["MLFLOW_ENABLE_DB_SDK"] = "true"

# # Avoid "more than one authorization method configured: basic and oauth" error in Databricks SDK
# if os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET"):
#     os.environ.pop("DATABRICKS_USERNAME", None)

# Clean and format DATABRICKS_HOST to ensure it has https:// scheme
db_host = os.environ.get("DATABRICKS_HOST", "")
if db_host and not db_host.startswith("http://") and not db_host.startswith("https://"):
    os.environ["DATABRICKS_HOST"] = f"https://{db_host}"

import mlflow
import requests
import gradio as gr
from google import genai
from google.genai import types

# =====================================================================
# 1. CONFIGURE MLFLOW TRACKING FOR DATABRICKS
# =====================================================================
# Tell MLflow to use the Databricks tracking server
mlflow.set_tracking_uri("databricks")

# In Databricks, experiments are represented by paths in the workspace.
# By default, we write to a Shared workspace path, but you can override this in your .env
# default_user = os.environ.get("DATABRICKS_USERNAME", "Shared")
experiment_path = os.environ.get("DATABRICKS_EXPERIMENT_PATH", f"/Users/{default_user}/Gemini-Gradio-Agent")

print(f"Configuring Databricks MLflow Tracking...")
print(f"Targeting Experiment Workspace Path: {os.environ.get('DATABRICKS_EXPERIMENT_PATH') or experiment_path}")

try:
    mlflow.set_experiment(experiment_path)
except Exception as e:
    print(f"⚠️ Warning: Could not set experiment path '{experiment_path}' directly. "
          f"Ensure the path exists or is in a workspace you have write access to. "
          f"Error: {e}")

# Enable automatic tracing for all Gemini API calls
mlflow.gemini.autolog()

# Initialize Gemini Client
client = genai.Client()

# =====================================================================
# 2. DEFINE AGENT TOOLS
# =====================================================================

def get_current_weather(city: str) -> str:
    """Get the current weather for a city using the Open-Meteo public API.
    
    Args:
        city: The name of the city to get the weather for.
    """
    print(f"\n[Tool Execution] get_current_weather called with city='{city}'")
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en"
        geo_response = requests.get(geocode_url, timeout=5)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"Error: City '{city}' could not be resolved to coordinates."
            
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        resolved_name = location["name"]
        country = location.get("country", "")
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url, timeout=5)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        current = weather_data.get("current_weather")
        if not current:
            return f"Error: Could not retrieve current weather conditions for {resolved_name}."
            
        temp = current["temperature"]
        windspeed = current["windspeed"]
        weather_code = current["weathercode"]
        
        conditions = {
            0: "Clear sky",
            1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            95: "Thunderstorm"
        }
        condition = conditions.get(weather_code, "Unknown conditions")
        
        return (f"The current weather in {resolved_name}, {country} is {condition} with a "
                f"temperature of {temp}°C (Wind speed: {windspeed} km/h).")
                
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return f"The weather in {city} is sunny and 22°C (fallback)."


def search_web(query: str) -> str:
    """Search the web for real-time general information using the Tavily Search API.
    
    Args:
        query: The search query to run on the web.
    """
    print(f"\n[Tool Execution] search_web called with query='{query}'")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True
            }
            response = requests.post(url, json=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
            if data.get("answer"):
                return data["answer"]
            results = data.get("results", [])
            snippets = [r["content"] for r in results[:3]]
            return "\n".join(snippets) if snippets else "No search results found."
        except Exception as e:
            return f"Error executing Tavily search: {e}"
    else:
        return f"Search Result (Simulated) for '{query}': Observability metrics are successfully configured."

# =====================================================================
# 3. GRADIO CHAT HANDLER WITH MLFLOW SPAN DECORATOR
# =====================================================================

sessions = {}

@mlflow.trace(name="gradio_agent_chat_turn")
def get_agent_response(message: str, session_id: str):
    """
    Traces the chat invocation using the @mlflow.trace decorator.
    Spans are exported to Databricks MLflow.
    """
    # Set custom trace attributes for user tracking
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "application.interface": "gradio",
            "environment": "databricks-tracking"
        }
    )
    
    if session_id not in sessions:
        sessions[session_id] = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=[get_current_weather, search_web],
                system_instruction=(
                    "You are a helpful assistant with real-time tools. Use get_current_weather "
                    "for weather queries, and search_web for general search queries. Be concise."
                )
            )
        )
    
    chat_session = sessions[session_id]
    response = chat_session.send_message(message)
    
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_inputs({"user_message": message})
        active_span.set_outputs({"agent_response": response.text})
        
    return response.text


def gradio_chat(message: str, history: list, request: gr.Request):
    session_id = request.session_hash if request else "default_session"
    response_text = get_agent_response(message, session_id)
    return response_text

# =====================================================================
# 4. LAUNCH GRADIO INTERFACE
# =====================================================================
demo = gr.ChatInterface(
    fn=gradio_chat,
    title="Gemini Multi-Tool Agent (Databricks Traced) 🕵️‍♂️",
    description="All prompts and tool invocations are traced to your Databricks MLflow workspace!",
    examples=["What is the weather in Tokyo?", "Who won the latest Super Bowl?"],
)

if __name__ == "__main__":
    # Ensure Databricks variables are configured
    databricks_host = os.environ.get("DATABRICKS_HOST")
    
    # Check for Service Principal or Token auth variables
    sp_auth = os.environ.get("DATABRICKS_CLIENT_ID") and os.environ.get("DATABRICKS_CLIENT_SECRET")
    token_auth = os.environ.get("DATABRICKS_TOKEN")
    
    if not databricks_host:
        print("❌ Error: DATABRICKS_HOST environment variable is not set in .env")
        exit(1)
        
    if not (sp_auth or token_auth):
        print("❌ Error: Databricks authentication requires EITHER Service Principal credentials "
              "(DATABRICKS_CLIENT_ID & DATABRICKS_CLIENT_SECRET) OR a personal token (DATABRICKS_TOKEN) in .env")
        exit(1)
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found. Please add it to your .env file.")
        exit(1)
        
    print("Launching Gradio UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
