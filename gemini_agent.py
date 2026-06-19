import os
import mlflow
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# =====================================================================
# 1. CONFIGURE MLFLOW TRACING FOR THE AGENT
# =====================================================================
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Gemini Multi-Tool Agent")
mlflow.gemini.autolog()


# =====================================================================
# 2. DEFINE THE TOOLS (FUNCTIONS)
# =====================================================================

def get_current_weather(city: str) -> str:
    """Get the current weather for a city using the Open-Meteo public API.
    
    Args:
        city: The name of the city to get the weather for.
    """
    print(f"\n[Tool Execution] get_current_weather called with city='{city}'")
    try:
        # Step 1: Resolve city name to coordinates (latitude/longitude) using free Geocoding API
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
        
        # Step 2: Fetch current weather for coordinates
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
        
        # Interpret standard WMO Weather Codes
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
        print(f"Error fetching live weather: {e}. Using simulated fallback.")
        return f"The weather in {city} is sunny and 72°F (fallback)."


def search_web(query: str) -> str:
    """Search the web for real-time general information using the Tavily Search API.
    
    Args:
        query: The search query to run on the web.
    """
    print(f"\n[Tool Execution] search_web called with query='{query}'")
    
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        print("Using live Tavily Search API...")
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
            
            # Return Tavily's generated summary if available
            if data.get("answer"):
                return data["answer"]
            
            # Fallback to returning the top 3 snippet results
            results = data.get("results", [])
            snippets = [r["content"] for r in results[:3]]
            return "\n".join(snippets) if snippets else "No search results found."
        except Exception as e:
            return f"Error executing Tavily search: {e}"
    else:
        print("TAVILY_API_KEY not found. Simulating search results...")
        q = query.lower()
        if "population" in q:
            return "According to search engines, the population of San Francisco is approximately 808,000."
        else:
            return f"Search Result (Simulated) for '{query}': Observability metrics are successfully configured."


# =====================================================================
# 3. INITIALIZE GEMINI CLIENT AND CREATE THE CHAT AGENT
# =====================================================================
client = genai.Client()

print("=== Initializing Gemini Multi-Tool Chat Agent ===")

# Create chat with both tools configured
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        tools=[get_current_weather, search_web],
        system_instruction=(
            "You are a helpful assistant with real-time tools. Use get_current_weather "
            "for weather queries, and search_web for general search queries. Be concise."
        )
    )
)


# =====================================================================
# 4. INVOKE THE AGENT
# =====================================================================
# The model will select get_current_weather for the weather query and
# search_web for the population query.
try:
    # Query 1: Weather (should trigger get_current_weather)
    weather_query = "What is the weather in Paris?"
    print(f"\nSending User Query: '{weather_query}'")
    response_1 = chat.send_message(weather_query)
    print("\n=== Agent Response 1 ===")
    print(response_1.text)

    # Query 2: Search (should trigger search_web)
    search_query = "What is the population of San Francisco?"
    print(f"\nSending User Query: '{search_query}'")
    response_2 = chat.send_message(search_query)
    print("\n=== Agent Response 2 ===")
    print(response_2.text)

except Exception as e:
    print(f"\n[Error] Live API invocation failed: {e}")
    print("Ensure GEMINI_API_KEY environment variable is configured.")
