import logging

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload
from griptape_nodes.retained_mode.events.workflow_events import (
    PublishWorkflowRegisteredEventData,
    PublishWorkflowRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.publish_workflow.local_publish_options import get_local_publish_options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes")


def _publish_workflow_request_handler(request: RequestPayload) -> ResultPayload:
    if not isinstance(request, PublishWorkflowRequest):
        msg = f"Expected PublishWorkflowRequest, got {type(request).__name__}"
        raise TypeError(msg)
    from griptape_nodes_library.publish_workflow.local_publisher import LocalPublisher

    publisher = LocalPublisher(
        workflow_name=request.workflow_name,
        metadata=request.metadata,
        pickle_control_flow_result=request.pickle_control_flow_result,
    )
    return publisher.publish_workflow()


class GriptapeNodesLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for the default Griptape Nodes Library."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:  # noqa: ARG002
        """Called after all nodes have been loaded from the library."""
        GriptapeNodes.LibraryManager().on_register_event_handler(
            request_type=PublishWorkflowRequest,
            handler=_publish_workflow_request_handler,
            library_data=library_data,
            event_data=PublishWorkflowRegisteredEventData(
                start_flow_node_type="StartFlow",
                start_flow_node_library_name=library_data.name,
                end_flow_node_type="EndFlow",
                end_flow_node_library_name=library_data.name,
                get_publish_options=get_local_publish_options,
            ),
        )
        # Auto-start local model server
        _start_model_server()


def _start_model_server() -> None:
    """Auto-start local model server when Griptape loads"""
    import subprocess
    import sys
    import os
    import threading

    # Find server path relative to this file or use environment variable
    server_dir = os.environ.get(
        "GRIPTAPE_MODEL_SERVER_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "model_server")
    )
    server_dir = os.path.normpath(server_dir)

    if sys.platform == "win32":
        server_python = os.path.join(server_dir, ".venv", "Scripts", "python.exe")
    else:
        server_python = os.path.join(server_dir, ".venv", "bin", "python")

    server_script = os.path.join(server_dir, "server.py")
    health_url = "http://127.0.0.1:8088/health"

    # Check if already running
    try:
        import urllib.request
        urllib.request.urlopen(health_url, timeout=2)
        logger.info("[ModelServer] Already running at http://127.0.0.1:8088")
        return
    except Exception:
        pass

    # Check files exist
    if not os.path.exists(server_python) or not os.path.exists(server_script):
        logger.warning(f"[ModelServer] Server not found at: {server_dir}")
        logger.warning("[ModelServer] Set GRIPTAPE_MODEL_SERVER_DIR env var or run install.py")
        return

    def run_server():
        try:
            logger.info(f"[ModelServer] Starting from: {server_dir}")
            kwargs = {"cwd": server_dir}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen([server_python, server_script], **kwargs)
            logger.info("[ModelServer] Started at http://127.0.0.1:8088")
        except Exception as e:
            logger.warning(f"[ModelServer] Failed to start: {e}")

    threading.Thread(target=run_server, daemon=True).start()
