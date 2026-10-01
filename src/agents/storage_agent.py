"""Definición del agente de persistencia de eventos y clips."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="storage",
    kind=AgentKind.TOOL,
    purpose="Persiste eventos y solicita clips al pipeline compartido.",
    singleton=True,
    consumes=(Topic.EVENTS,),
    publishes=(Topic.HEALTH,),
)
