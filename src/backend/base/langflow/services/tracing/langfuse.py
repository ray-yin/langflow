from __future__ import annotations

import os
import traceback
import types
from collections import OrderedDict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger
from typing_extensions import override

from langflow.serialization.serialization import serialize
from langflow.services.tracing.base import BaseTracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from langchain.callbacks.base import BaseCallbackHandler

    from langflow.graph.vertex.base import Vertex
    from langflow.services.tracing.schema import Log


class LangFuseTracer(BaseTracer):
    flow_id: str

    def __init__(
        self,
        trace_name: str,
        trace_type: str,
        project_name: str,
        trace_id: UUID,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.project_name = project_name
        self.trace_name = trace_name
        self.trace_type = trace_type
        self.trace_id = trace_id
        self.user_id = user_id
        self.session_id = session_id
        self.flow_id = trace_name.split(" - ")[-1]
        self.spans: dict = OrderedDict()  # spans that are not ended

        config = self._get_config()
        self._ready: bool = self.setup_langfuse(config) if config else False

    @property
    def ready(self):
        return self._ready

    def setup_langfuse(self, config) -> bool:
        try:
            from langfuse import Langfuse

            self._client = Langfuse(**config)
            try:
                from langfuse.api.core.request_options import RequestOptions

                self._client.client.health.health(request_options=RequestOptions(timeout_in_seconds=1))
            except Exception as e:  # noqa: BLE001
                logger.debug(f"can not connect to Langfuse: {e}")
                return False
            self.trace = self._client.trace(
                id=str(self.trace_id),
                name=self.flow_id,
                user_id=self.user_id,
                session_id=self.session_id,
            )

        except ImportError:
            logger.exception("Could not import langfuse. Please install it with `pip install langfuse`.")
            return False

        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error setting up LangSmith tracer: {e}")
            return False

        return True

    def _convert_value_to_safe_type(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._convert_value_to_safe_type(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._convert_value_to_safe_type(item) for item in value]
        if isinstance(value, types.GeneratorType):
            # If a generator is passed directly, convert its representation to a string.
            # The component itself (e.g., ChatOutput) should be responsible for
            # consuming the generator and providing the materialized content for tracing.
            logger.warning(
                f"LangfuseTracer: Encountered a raw generator object: {value}. "
                f"Converting to string representation. For full content tracing, "
                f"the component should materialize the generator's output before tracing."
            )
            return str(value)
        return value

    def _convert_dict_to_safe_types(self, data: dict[str, Any] | None) -> dict[str, Any] | None:
        if data is None:
            return None
        return {str(k): self._convert_value_to_safe_type(v) for k, v in data.items()}

    @override
    def add_trace(
        self,
        trace_id: str,  # actualy component id
        trace_name: str,
        trace_type: str,
        inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        vertex: Vertex | None = None,
    ) -> None:
        start_time = datetime.now(tz=timezone.utc)
        if not self._ready:
            return

        processed_inputs = self._convert_dict_to_safe_types(inputs)

        final_metadata: dict[str, Any] = {"from_langflow_component": True, "component_id": trace_id}
        if trace_type:
            final_metadata["trace_type"] = trace_type
        if metadata:
            safe_user_metadata = self._convert_dict_to_safe_types(metadata)
            if safe_user_metadata:
                final_metadata.update(safe_user_metadata)

        name = trace_name.removesuffix(f" ({trace_id})")
        content_span = {
            "name": name,
            "input": processed_inputs,
            "metadata": final_metadata,
            "start_time": start_time,
        }

        # if two component is built concurrently, will use wrong last span. just flatten now, maybe fix in future.
        # if len(self.spans) > 0:
        #     last_span = next(reversed(self.spans))
        #     span = self.spans[last_span].span(**content_span)
        # else:
        span = self.trace.span(**serialize(content_span))

        self.spans[trace_id] = span

    @override
    def end_trace(
        self,
        trace_id: str,
        trace_name: str,
        outputs: dict[str, Any] | None = None,
        error: Exception | None = None,
        logs: Sequence[Log | dict] = (),
    ) -> None:
        end_time = datetime.now(tz=timezone.utc)
        if not self._ready:
            return

        span = self.spans.pop(trace_id, None)
        if span:
            processed_outputs = self._convert_dict_to_safe_types(outputs)

            safe_logs = []
            if logs:
                for log_entry in logs:
                    if isinstance(log_entry, dict):
                        safe_logs.append(self._convert_dict_to_safe_types(log_entry))
                    else:  # Log object
                        # Assuming Log object is inherently serializable or has a method like model_dump()
                        safe_logs.append(log_entry)

            # This dictionary becomes the value for the "output" key in the span update payload
            output_field_for_span_update: dict = {}
            if processed_outputs:
                output_field_for_span_update.update(processed_outputs)

            if error:  # Include error string in the "output" field as per original logic
                output_field_for_span_update["error"] = str(error)

            if safe_logs:
                output_field_for_span_update["logs"] = list(safe_logs)

            content_for_update: dict[str, Any] = {"end_time": end_time}
            if output_field_for_span_update:  # Only add "output" key if there's content for it
                content_for_update["output"] = output_field_for_span_update

            if error:  # Also set span-level error indicators
                content_for_update["level"] = "ERROR"
                string_stacktrace = traceback.format_exception(error)
                error_message = f"{error.__class__.__name__}: {error}\n\n{''.join(string_stacktrace)}"
                content_for_update["status_message"] = error_message
            span.update(**serialize(content_for_update))

    @override
    def end(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        error: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ready:
            return

        safe_inputs = self._convert_dict_to_safe_types(inputs)
        safe_outputs = self._convert_dict_to_safe_types(outputs)
        safe_metadata = self._convert_dict_to_safe_types(metadata)

        update_payload: dict[str, Any] = {}
        if safe_inputs is not None:
            update_payload["input"] = safe_inputs
        if safe_outputs is not None:
            update_payload["output"] = safe_outputs
        if safe_metadata is not None:
            update_payload["metadata"] = safe_metadata

        serialized_payload = serialize(update_payload)

        if error:
            serialized_payload["level"] = "ERROR"
            string_stacktrace = traceback.format_exception(error)
            error_message = f"{error.__class__.__name__}: {error}\n\n{''.join(string_stacktrace)}"
            serialized_payload["status_message"] = error_message

        self.trace.update(**serialized_payload)

    def get_langchain_callback(self) -> BaseCallbackHandler | None:
        if not self._ready:
            return None

        # get callback from parent span
        stateful_client = self.spans[next(reversed(self.spans))] if len(self.spans) > 0 else self.trace
        return stateful_client.get_langchain_handler()

    @staticmethod
    def _get_config() -> dict:
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", None)
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", None)
        host = os.getenv("LANGFUSE_HOST", None)
        if secret_key and public_key and host:
            return {"secret_key": secret_key, "public_key": public_key, "host": host}
        return {}
