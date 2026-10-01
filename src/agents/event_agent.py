"""Definición del agente de reglas y eventos."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="event",
    kind=AgentKind.REACT,
    purpose="Evalúa zonas, merodeo y conteo sobre tracks para emitir eventos.",
    singleton=True,
    consumes=(Topic.TRACKS,),
    publishes=(Topic.EVENTS, Topic.HEALTH),
)
