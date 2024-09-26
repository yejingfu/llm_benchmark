from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry import trace, baggage

from opentelemetry.sdk.resources import Resource, SERVICE_NAME, HOST_NAME
from opentelemetry.context.context import Context
from opentelemetry.sdk.environment_variables import (OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
from opentelemetry.semconv_ai import SpanAttributes as BaseSpanAttributes
from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import (TraceContextTextMapPropagator)

import grpc
import os
import traceback

def inner_method():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("child_span") as child_span:
        print("hello world")

def outer_method():
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("parent_span") as parent_span:
        inner_method()

def baggage_and_attribute_usage():
    tracer = trace.get_tracer(__name__)
    global_ctx = baggage.set_baggage("key", "value_from_global_ctx")
    with tracer.start_as_current_span(name='baggage_parent_span', attributes={'attribute_key': 'value'}) as baggage_parent_span:
        parent_ctx = baggage.set_baggage("key", "value_from_parent_ctx")
        with tracer.start_as_current_span(name='baggage_child_span', context=parent_ctx) as baggage_child_span:
            child_ctx = baggage.set_baggage("key", "value_from_child_ctx")
            print(baggage.get_baggage("key", child_ctx))
            print(baggage.get_baggage("key", parent_ctx))
    print(baggage.get_baggage("key", global_ctx))

def _load_tls_secret(tls_config_val: str) -> bytes:
    """If the config value points at a file, load it, otherwise assume it's an inline string."""
    if os.path.exists(tls_config_val):
        with open(tls_config_val, "rb") as handle:
            return handle.read()
    else:
        print(f"Failed to load otel tls secrect from {tls_config_val}")
        return tls_config_val.encode("utf-8")

def init_opentelemetry():
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    assert protocol == "grpc", "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL != grpc"
    otlp_traces_endpoint = os.getenv("OTEL_TRACE_ENDPOINT", "otel-public.aicloud.pplabs.tech:14317")
    print(f"ENV[OTEL_TRACE_ENDPOINT]: {otlp_traces_endpoint}")
    tls_secrect_folder=os.getenv("OTEL_TLS_DIR")
    if not tls_secrect_folder:
        if "aicloud.pplabs.tech" in otlp_traces_endpoint:
            secret_key_folder = "/vllm-workspace/otel/pplabs"
        elif "aicloud.paigod.work" in otlp_traces_endpoint:
            secret_key_folder = "/vllm-workspace/otel/paigod"
        else:
            raise ValueError(f"Invalid OTEL traces endpoint: {otlp_traces_endpoint}")
    print(f"ENV[OTEL_TLS_DIR]: {secret_key_folder}")
    creds_kwargs = {
        "root_certificates": _load_tls_secret(secret_key_folder + "/ca.pem"),
        "private_key": _load_tls_secret(secret_key_folder + "/client-key.pem"),
        "certificate_chain": _load_tls_secret(secret_key_folder + "/client-chain.pem")
    }
    ssl_credentials = grpc.ssl_channel_credentials(**creds_kwargs)
    otel_token = os.environ.get("OTEL_TOKEN", "tQPpDthXQCbLShFmymo6Mo4pHvcZr10A")
    print(f"ENV[OTEL_TOKEN]: {otel_token}")

    def metadata_callback(context, callback):
        #headers = (("authorization", "Bearer tQPpDthXQCbLShFmymo6Mo4pHvcZr10A"),)
        headers = (("authorization", "Bearer " + otel_token),)
        callback(headers, None)
    auth_credentials = grpc.metadata_call_credentials(metadata_callback)
    composite_credentials = grpc.composite_channel_credentials(ssl_credentials, auth_credentials)

    exporter_kwargs = {
        "endpoint": otlp_traces_endpoint,
        "credentials": composite_credentials
    }
    resource = Resource(attributes={
        SERVICE_NAME: "vllm.test",
        HOST_NAME: "vllm.localhost",
        "vllm.version": "internal"
    })
    span_processor = BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs))
    trace_provider = TracerProvider(resource=resource, active_span_processor=span_processor)
    trace.set_tracer_provider(trace_provider)
    #set_tracer_provider(trace_provider)
    #tracer = trace_provider.get_tracer(instrumenting_module_name)

if __name__ == '__main__':
    try:
        init_opentelemetry()
        outer_method()
        baggage_and_attribute_usage()
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")


