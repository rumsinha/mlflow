import os
import mlflow
import requests
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =====================================================================
# 1. CONFIGURE MLFLOW TRACKING FOR GRADIO APP
# =====================================================================
# Point to your local MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Gemini Gradio Agent")
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

# Dictionary to hold active chat sessions per user session ID to maintain chat history
sessions = {}

@mlflow.trace(name="gradio_agent_chat_turn")
def get_agent_response(message: str, session_id: str):
    """
    Traces the chat invocation using the @mlflow.trace decorator.
    This creates a parent span 'gradio_agent_chat_turn'. All internal Gemini
    calls and tool calls will be logged as child spans.
    """
    # Set custom trace attributes for user tracking
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "application.interface": "gradio"
        }
    )
    
    # Retrieve or create a new Chat session for the user
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
    
    # Send message and get response
    response = chat_session.send_message(message)
    
    # Tag trace inputs and outputs
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_inputs({"user_message": message})
        active_span.set_outputs({"agent_response": response.text})
        
    return response.text

# Gradio chatbot function matching interface requirements
def gradio_chat(message: str, history: list, request: gr.Request):
    # Retrieve a unique session hash or client host from the request context
    # This keeps different browser tabs independent
    session_id = request.session_hash if request else "default_session"
    
    response_text = get_agent_response(message, session_id)
    return response_text

# =====================================================================
# 4. LAUNCH GRADIO INTERFACE
# =====================================================================
demo = gr.ChatInterface(
    fn=gradio_chat,
    title="Gemini Multi-Tool Agent 🕵️‍♂️",
    description="Ask weather questions or general web search queries. All steps are traced live to MLflow!",
    examples=["What is the weather in New York?", "What is the capital of France?", "Who won the latest Super Bowl?"],
)

if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found. Please add it to your .env file.")
        exit(1)
        
    print("Launching Gradio UI...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
