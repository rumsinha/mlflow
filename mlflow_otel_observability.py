import os
import time
import mlflow
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenTelemetry API to show how MLflow tracing interacts with OTel under the hood
from opentelemetry import trace as otel_trace

# =====================================================================
# 1. CONFIGURE MLFLOW TRACKING
# =====================================================================
# Tell MLflow where the tracking server is located (our local server running on port 5001)
mlflow.set_tracking_uri("http://localhost:5001")

# Tell MLflow which experiment to log these traces to
EXPERIMENT_NAME = "Gemini Observability Demo"
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable automatic tracing for all Gemini API calls using the google-genai SDK.
# Under the hood, MLflow hooks into the Gemini API requests using OpenTelemetry.
mlflow.gemini.autolog()


# =====================================================================
# 2. DEFINE THE CHAT COMPLETION FUNCTION WITH USER & SESSION TRACKING
# =====================================================================
@mlflow.trace(name="chat_completion_orchestrator")
def chat_completion(prompt: str, user_id: str, session_id: str):
    """
    Processes a chat message with user and session tracking.
    
    This function is decorated with @mlflow.trace, which tells MLflow to create a parent span
    for this execution. Inside, we can add metadata, logs, and sub-spans.
    """
    print(f"\n[Trace Start] User: {user_id} | Session: {session_id} | Prompt: '{prompt}'")
    
    # Add user and session context to the current trace.
    # MLflow reads "mlflow.trace.user" and "mlflow.trace.session" to categorize and index
    # traces in the MLflow UI, allowing developers to filter and analyze traces per user or session.
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "application.environment": "development",
        }
    )

    # Perform a preprocessing step in a custom sub-span to show nesting
    preprocessed_prompt = preprocess_input(prompt)

    # Check if Gemini API Key is available.
    # If it is, we make a live call. If not, we simulate the LLM call so that the script runs successfully
    # and generates traces on the local MLflow server anyway.
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if api_key:
        print("Using live Gemini API...")
        client = genai.Client()
        # The following call is automatically instrumented by mlflow.gemini.autolog(),
        # producing a child span representing the model inference.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=preprocessed_prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful and brief AI assistant."
            )
        )
        response_text = response.text
    else:
        print("GEMINI_API_KEY not found. Simulating LLM call to demonstrate tracing flow...")
        response_text = simulate_gemini_call(preprocessed_prompt)

    # Post-process response in another child span
    final_output = postprocess_output(response_text)
    
    # Add attributes directly to the parent orchestrator span
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_inputs({"prompt": prompt, "user_id": user_id, "session_id": session_id})
        active_span.set_outputs({"response": final_output})

    return final_output


@mlflow.trace(name="preprocess_input")
def preprocess_input(text: str) -> str:
    """Simulates a preprocessing step in its own span."""
    time.sleep(0.1)  # Simulate some processing latency
    return text.strip()


@mlflow.trace(name="simulate_gemini_call")
def simulate_gemini_call(prompt: str) -> str:
    """Simulates Gemini LLM inference and registers a child span."""
    time.sleep(0.4)  # Simulate model response latency (400ms)
    
    # Under the hood, we can manually simulate LLM span attributes:
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_attribute("llm.model_name", "gemini-2.5-flash")
        active_span.set_attribute("llm.prompt_tokens", len(prompt.split()))
        active_span.set_attribute("llm.completion_tokens", 15)
        active_span.set_attribute("llm.total_tokens", len(prompt.split()) + 15)
        
    return f"Simulated response to: '{prompt}'. Observability is working!"


@mlflow.trace(name="postprocess_output")
def postprocess_output(text: str) -> str:
    """Simulates a postprocessing step in its own span."""
    time.sleep(0.05)  # Simulate postprocessing latency
    return f"[Formatted] {text}"


# =====================================================================
# 3. DIRECT OPENTELEMETRY SDK INTEGRATION
# =====================================================================
# MLflow tracing uses standard OpenTelemetry APIs. We can get the tracer from the
# global trace provider configured by MLflow and create spans using standard OTel APIs.
def demonstrate_otel_span_nesting():
    print("\nDemonstrating direct OpenTelemetry SDK spans linked with MLflow...")
    
    # Get the global tracer configured by MLflow's tracing system
    tracer = otel_trace.get_tracer("mlflow.observability.demo")
    
    # Start a span using standard OpenTelemetry SDK
    with tracer.start_as_current_span("native_otel_parent_span") as otel_span:
        otel_span.set_attribute("custom.otel.attribute", "demo-value")
        
        # We can perform a traced operation inside
        with tracer.start_as_current_span("native_otel_child_span") as child_span:
            child_span.set_attribute("sub_task.status", "success")
            time.sleep(0.1)
            print("Native OpenTelemetry spans created successfully!")


if __name__ == "__main__":
    # Run chat completions to simulate a conversation session
    print("=== Running Traced LLM Application ===")
    
    # Session 1: User Alice asking two questions
    chat_completion(
        prompt="Explain the difference between a Trace and a Span in 1 sentence.",
        user_id="user_alice_123",
        session_id="session_chat_abc"
    )
    
    chat_completion(
        prompt="How does MLflow help with tracing LLM applications?",
        user_id="user_alice_123",
        session_id="session_chat_abc"
    )

    # Session 2: User Bob asking a question
    chat_completion(
        prompt="Tell me a joke about software developers.",
        user_id="user_bob_456",
        session_id="session_chat_xyz"
    )

    # Demonstrate native OpenTelemetry tracing
    demonstrate_otel_span_nesting()

    print("\n=== Application Run Finished ===")
    print("You can view the traces in the MLflow UI at: http://127.0.0.1:5001")
