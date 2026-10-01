"""Definición del puente de comunicaciones externas."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="comms",
    kind=AgentKind.TOOL,
    purpose="Expone eventos y salud mediante adaptadores HTTP o WebSocket.",
    singleton=True,
    consumes=(Topic.EVENTS, Topic.HEALTH),
    publishes=(),
)
