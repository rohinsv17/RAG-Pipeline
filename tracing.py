"""
Rohin Vijay
Tracing file for RAG
--------------------------------
Allows observability of the RAG pipeline by registering a tracerprovider
Usage: initTracing() imported into query.py, tracing visible at dashboard
Have "python -m phoenix.server.main serve" running in open terminal for Phoenix dashboard visibility
"""

from phoenix.otel import register


# Initializes tracing for the RAG pipeline
def initTracing():
    # Registers OpenTelemetry tracerprovider with Arize Phoenix
    tracer_provider = register(project_name = "rag-pipeline", auto_instrument = True, endpoint = "http://localhost:6006/v1/traces", protocol = "http/protobuf")

    print("[Tracing] Phoenix tracing initialized - http://localhost:6006\n")
    return tracer_provider