"""
tracing.py — Phoenix tracing setup
------------------------------------
Initializes Arize Phoenix tracing for the RAG pipeline.
Import and call initTracing() once at the start of any
file you want to trace (query.py, ingestion.py, etc.)

Keep Phoenix running in a separate terminal before using:
    python -m phoenix.server.main serve
"""

from phoenix.otel import register

# ─────────────────────────────────────────────
# TRACING SETUP
# ─────────────────────────────────────────────

def initTracing():
    # Registers Phoenix as the global OpenTelemetry tracer provider
    # auto_instrument=True automatically captures OpenAI SDK calls (LLM request/response)
    # endpoint points to the local Phoenix server started in the terminal
    tracer_provider = register(project_name = "rag-pipeline", auto_instrument = True, endpoint = "http://localhost:6006/v1/traces", protocol = "http/protobuf")

    print("[Tracing] Phoenix tracing initialized — http://localhost:6006\n")
    return tracer_provider