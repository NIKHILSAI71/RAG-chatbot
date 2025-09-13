from __future__ import annotations
from ragchat.core.config import settings

_inited = False

def init_tracing(app=None):
    global _inited
    if _inited or not settings.tracing_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

        resource = Resource.create({"service.name": settings.service_name})
        sampler = ParentBased(TraceIdRatioBased(settings.trace_sample_ratio))
        provider = TracerProvider(resource=resource, sampler=sampler)
        trace.set_tracer_provider(provider)
        if settings.otlp_endpoint:
            exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()
        _inited = True
    except Exception as e:
        print(f"[tracing] init failed: {e}")
