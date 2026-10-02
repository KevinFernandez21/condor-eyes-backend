"""Exportación opcional de decisiones a Langfuse (solo metadata filtrada).

Estado real de los agentes: ``event`` y ``supervisor`` son hoy **manejadores de
reglas** (``agents.handlers``), no agentes ReAct con LLM, aunque su ruta declare
``AgentKind.REACT``. Por eso aquí se trazan como *spans* de decisión por reglas
(``decision_engine="rules"``, ``llm=False``). Cuando existan agentes ReAct, las
llamadas al modelo se añadirán como observaciones ``generation`` hijas del span
de decisión; este módulo no inventa generaciones que no ocurren.

Qué se traza (una traza de Langfuse por mensaje, con ``trace_id`` determinista
derivado del ``event_id``, así una reentrega no duplica):

- ``fusion.decision``: decisión de fusión (``payload.decision_id``), con un span
  hijo por referencia de evidencia.
- ``event.rule``: evento emitido por el rol ``event``.
- ``supervisor.command``: comando emitido por el rol ``supervisor``.

Todo pasa antes por ``tracing.privacy``. Las claves salen **solo del entorno**
(``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` y ``LANGFUSE_HOST`` o
``LANGFUSE_BASE_URL``). Sin ellas el tracer no se crea y el sistema corre igual.
El host es obligatorio a propósito: el SDK usaría Langfuse Cloud por defecto y
eso sacaría metadata del sitio sin que nadie lo haya decidido.

El SDK ``langfuse`` es un extra opcional (``uv sync --extra observability``) y
se importa de forma diferida.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bus import Topic

from .privacy import PrivacyFilter
from .store import TraceStore

logger = logging.getLogger(__name__)

ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_HOST = "LANGFUSE_HOST"
ENV_BASE_URL = "LANGFUSE_BASE_URL"


@dataclass(frozen=True)
class LangfuseConfig:
    public_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    host: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LangfuseConfig | None:
        """Lee la configuración del entorno; ``None`` si falta algo (desactivado)."""
        env = os.environ if env is None else env
        public = env.get(ENV_PUBLIC_KEY, "").strip()
        secret = env.get(ENV_SECRET_KEY, "").strip()
        host = (env.get(ENV_BASE_URL) or env.get(ENV_HOST) or "").strip()
        if not (public and secret and host):
            return None
        return cls(public, secret, host)


def _make_client(config: LangfuseConfig) -> Any:
    from langfuse import Langfuse  # diferido: extra opcional

    return Langfuse(
        public_key=config.public_key, secret_key=config.secret_key, base_url=config.host
    )


def classify(topic: Topic, data: Mapping[str, Any]) -> str | None:
    """Nombre del tipo de decisión que representa el mensaje, o ``None``."""
    if topic is Topic.EVENTS:
        if "decision_id" in (data.get("payload") or {}):
            return "fusion.decision"
        if data.get("source") == "event":
            return "event.rule"
    elif topic is Topic.COMMANDS and data.get("source") == "supervisor":
        return "supervisor.command"
    return None


_DEDUPE_SIZE = 4096


class LangfuseDecisionTracer:
    """Oyente del tap que exporta decisiones a un cliente Langfuse.

    ``on_message`` es síncrono y no lanza: el SDK encola y envía en segundo
    plano, y cualquier fallo se registra sin tocar el bus.
    """

    def __init__(
        self,
        client: Any,
        store: TraceStore,
        privacy: PrivacyFilter | None = None,
        dedupe_size: int = _DEDUPE_SIZE,
    ) -> None:
        self._client = client
        self._store = store
        self._privacy = privacy or PrivacyFilter.from_env()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._dedupe_size = max(1, dedupe_size)
        self._warned = False

    def on_message(self, topic: Topic, data: dict[str, Any]) -> None:
        name = classify(Topic(topic), data)
        if name is None:
            return
        event_id = str(data.get("event_id"))
        if event_id in self._seen:  # reentrega: el mismo mensaje no se exporta dos veces
            return
        self._seen[event_id] = None
        while len(self._seen) > self._dedupe_size:
            self._seen.popitem(last=False)
        try:
            self._export(name, data)
        except Exception:
            if not self._warned:  # una vez: no inundar el log si Langfuse está caído
                self._warned = True
                logger.warning("Langfuse no pudo registrar '%s'; se ignora", name, exc_info=True)

    def _export(self, name: str, data: dict[str, Any]) -> None:
        payload = self._privacy.payload(data.get("payload") or {})
        evidence = payload.pop("evidence", [])
        trace = self._store.get(data.get("correlation_id") or data["event_id"]) or {}
        chain = [self._privacy.hop(hop) for hop in trace.get("hops", [])]
        end_to_end = trace.get("end_to_end_ms")
        finite = isinstance(end_to_end, int | float) and math.isfinite(end_to_end)
        output = {**payload, "end_to_end_ms": end_to_end if finite else None}
        envelope = self._privacy.envelope({**data, "payload": {}})
        metadata = {
            "decision_engine": "fusion_rules" if name == "fusion.decision" else "rules",
            "llm": False,
            "topic": envelope["topic"],
            "source": envelope["source"],
            "event_id": envelope["event_id"],
            "correlation_id": envelope["correlation_id"],
        }
        trace_id = self._client.create_trace_id(seed=envelope["event_id"])
        span = self._client.start_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="span",
            input={"chain": chain},
            output=output,
            metadata=metadata,
        )
        try:
            for ref in evidence:
                child = span.start_observation(
                    name=f"evidence.{ref.get('kind', 'unknown')}", as_type="span", input=ref
                )
                child.end()
        finally:
            span.end()

    def close(self) -> None:
        """Vacía la cola y apaga el cliente (idempotente en la práctica)."""
        try:
            self._client.flush()
            self._client.shutdown()
        except Exception:
            logger.warning("Langfuse no cerró limpiamente", exc_info=True)


def create_tracer_from_env(
    store: TraceStore, env: Mapping[str, str] | None = None
) -> LangfuseDecisionTracer | None:
    """Tracer activo solo si el entorno lo configura y el SDK está instalado."""
    config = LangfuseConfig.from_env(env)
    if config is None:
        return None
    # Antes de crear el cliente: una clave HMAC inválida debe fallar fuerte (PrivacyConfigError).
    privacy = PrivacyFilter.from_env(env)
    try:
        client = _make_client(config)
    except ImportError:
        logger.warning(
            "Langfuse está configurado pero el SDK no está instalado "
            "(uv sync --extra observability); trazado externo desactivado"
        )
        return None
    except Exception:
        logger.warning("No se pudo crear el cliente de Langfuse; desactivado", exc_info=True)
        return None
    logger.info("Langfuse activo: metadata filtrada hacia %s", config.host)
    return LangfuseDecisionTracer(client, store, privacy)
