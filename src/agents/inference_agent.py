"""Definición del único agente de inferencia compartida."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="inference",
    kind=AgentKind.TOOL,
    purpose="Controla el engine TensorRT compartido y publica detecciones como metadata.",
    singleton=True,
    consumes=(Topic.STREAM_STATUS, Topic.COMMANDS),
    publishes=(Topic.DETECTIONS, Topic.HEALTH),
)
