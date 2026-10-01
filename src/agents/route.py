"""Registro canónico de la ruta multiagente."""

from .base import AgentRoute
from .comms_agent import ROUTE as COMMS_ROUTE
from .event_agent import ROUTE as EVENT_ROUTE
from .inference_agent import ROUTE as INFERENCE_ROUTE
from .ingest_agent import ROUTE as INGEST_ROUTE
from .storage_agent import ROUTE as STORAGE_ROUTE
from .supervisor_agent import ROUTE as SUPERVISOR_ROUTE
from .tracker_agent import ROUTE as TRACKER_ROUTE

MULTIAGENT_ROUTE: tuple[AgentRoute, ...] = (
    INGEST_ROUTE,
    INFERENCE_ROUTE,
    TRACKER_ROUTE,
    EVENT_ROUTE,
    STORAGE_ROUTE,
    COMMS_ROUTE,
    SUPERVISOR_ROUTE,
)


def get_route(name: str) -> AgentRoute:
    """Obtiene la definición de un agente por nombre."""
    for route in MULTIAGENT_ROUTE:
        if route.name == name:
            return route
    raise KeyError(f"Agente desconocido: {name}")


def validate_route(routes: tuple[AgentRoute, ...] = MULTIAGENT_ROUTE) -> None:
    """Valida invariantes mínimas de la arquitectura multiagente."""
    names = [route.name for route in routes]
    if len(names) != len(set(names)):
        raise ValueError("La ruta contiene nombres de agente duplicados")

    inference = [route for route in routes if route.name == "inference"]
    if len(inference) != 1 or not inference[0].singleton:
        raise ValueError("La ruta debe tener un único agente de inferencia")

    ingest = [route for route in routes if route.name == "ingest"]
    if len(ingest) != 1 or ingest[0].singleton:
        raise ValueError("El agente de ingesta debe poder escalar por cámara")
