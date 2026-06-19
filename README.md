# GenAI Observability: Tracing Gemini Agents with MLflow & OpenTelemetry

This repository contains clean, production-grade example implementations demonstrating how to build and monitor **Google Gemini** applications and agentic workflows using **MLflow Tracing** and **OpenTelemetry (OTel)**.

By leveraging OTel under the hood, these setups provide complete transparency into nested LLM execution paths, token counts, system latencies, and tool calls (function calling) across local server databases and managed Databricks workspaces.

---

## 🚀 Key Features

* **Zero-Boilerplate Autologging:** Capture raw prompts, system instructions, generation settings, and token metrics automatically via `mlflow.gemini.autolog()`.
* **Hierarchical Custom Spans:** Use the `@mlflow.trace` decorator to wrap application logic, preprocessing functions, and custom postprocessors into clear parent-child trace trees.
* **Multi-Tool Agent Tracking:** Trace execution steps of agents that dynamically choose, call, and reflection-loop through Python tool functions (e.g. weather fetching).
* **Native OpenTelemetry Bridge:** Seamlessly mix standard OTel SDK instrumentation alongside high-level MLflow APIs in a unified telemetry pipeline.
* **Traced Interactive Web UIs:** An interactive chatbot frontend built with **Gradio** where user sessions and individual conversation turns are tracked end-to-end.
* **Dual Exporter Support:** Deploy locally to a SQLite backend or stream traces remotely to Databricks Experiments via secure OAuth/PAT authorization.

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| [gemini_agent.py] | A CLI agent utilizing `google-genai` and local MLflow tracing to dynamically execute weather and web search tools. |
| [gemini_gradio_agent.py](| An interactive Gradio web application logging chat turn telemetry locally. |
| [gemini_gradio_agentv2.py] | An enterprise-ready version of the Gradio agent configured to trace sessions to a managed Databricks cloud workspace. |
| [gemini_tracing.py] | A simple script demonstrating standard autologged Gemini API calls. |
| [gemini_workflows.py] | Multi-step workflows showcasing custom helper spans and hybrid LangChain integrations. |
| [mlflow_otel_observability.py]| Deep-dive observability pipeline illustrating session metadata tagging and native OTel SDK span nesting. |

---

## ⚙️ Getting Started

### 1. Installation
Install the required packages in your Python virtual environment:
```bash
pip install mlflow google-genai gradio opentelemetry-api opentelemetry-sdk requests python-dotenv
```

### 2. Environment Configuration
Create a `.env` file in the root of the repository:
```env
# Required Google Gemini API Key
GEMINI_API_KEY="your-gemini-api-key"

# (Optional) Web Search Tool API Key
TAVILY_API_KEY="your-tavily-search-key"

# (Optional) Remote Databricks Tracking Configuration
DATABRICKS_HOST="https://your-workspace-url.cloud.databricks.com"
DATABRICKS_TOKEN="your-personal-access-token"
DATABRICKS_EXPERIMENT_PATH="/Users/your_user_email/Gemini-Gradio-Agent"
```

### 3. Launching Local Observability Backends
To track telemetry locally, start the local MLflow Tracking Server:
```bash
mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow.db
```
Open **`http://localhost:5001`** in your browser to inspect the MLflow UI.

### 4. Running Scripts

* **To run the CLI agent:**
  ```bash
  python gemini_agent.py
  ```
* **To run the local Gradio chatbot:**
  ```bash
  python gemini_gradio_agent.py
  ```
  Then visit **`http://127.0.0.1:7860`** to chat with the agent and view session traces update in the MLflow dashboard under the *Traces* tab.
