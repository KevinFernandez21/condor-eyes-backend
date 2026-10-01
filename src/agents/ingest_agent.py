"""Definición del agente de ingesta, uno por cámara."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="ingest",
    kind=AgentKind.TOOL,
    purpose="Controla una fuente RTSP o USB y publica únicamente su estado.",
    singleton=False,
    consumes=(Topic.COMMANDS,),
    publishes=(Topic.STREAM_STATUS, Topic.HEALTH),
)
