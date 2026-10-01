"""Definición del agente de tracking por stream."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="tracker",
    kind=AgentKind.TOOL,
    purpose="Convierte detecciones en tracks por cámara sin transportar frames.",
    singleton=True,
    consumes=(Topic.DETECTIONS,),
    publishes=(Topic.TRACKS, Topic.HEALTH),
)
