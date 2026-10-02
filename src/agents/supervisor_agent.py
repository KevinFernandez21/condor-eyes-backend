"""Definición del agente supervisor."""

from bus.hub import Topic

from .base import AgentKind, AgentRoute

ROUTE = AgentRoute(
    name="supervisor",
    kind=AgentKind.REACT,
    purpose="Observa la salud del sistema y emite comandos operativos.",
    singleton=True,
    consumes=(Topic.HEALTH, Topic.STREAM_STATUS, Topic.ERRORS),
    publishes=(Topic.COMMANDS,),
)
