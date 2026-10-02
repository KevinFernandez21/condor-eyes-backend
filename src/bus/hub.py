"""Contrato tipado que aísla AgentScope del dominio de Condor Eye.

Este módulo transporta exclusivamente metadata serializable. Los frames y
buffers NVMM pertenecen al pipeline compartido y no forman parte del bus.

Define el vocabulario del bus (``Topic``), el envelope versionado con
identificadores de evento y correlación, los eventos de error y la validación
de tópicos y versiones. No contiene reglas de negocio ni lógica de transporte:
los adaptadores viven en ``bus.memory`` y ``bus.agentscope_hub``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

SCHEMA_VERSION = 1
"""Versión del esquema del envelope (campos comunes a todos los mensajes)."""

SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({SCHEMA_VERSION})
"""Versiones del envelope que este proceso sabe interpretar."""

DEFAULT_PAYLOAD_VERSION = 1
"""Versión de payload por defecto para cualquier tópico."""


class Topic(StrEnum):
    """Canales permitidos para la coordinación entre agentes."""

    STREAM_STATUS = "stream.status"
    DETECTIONS = "vision.detections"
    TRACKS = "vision.tracks"
    EVENTS = "events"
    HEALTH = "system.health"
    COMMANDS = "system.commands"
    ERRORS = "system.errors"


class HubError(Exception):
    """Error base del bus de metadata."""


class InvalidTopicError(HubError, ValueError):
    """El tópico no pertenece al vocabulario permitido."""


class UnsupportedVersionError(HubError, ValueError):
    """La versión de esquema o de payload no está soportada."""


class InvalidEnvelopeError(HubError, ValueError):
    """El envelope viola el contrato (campos vacíos, binarios, no serializable)."""


@dataclass(frozen=True, slots=True)
class MetadataEnvelope:
    """Mensaje serializable que nunca contiene frames de video.

    Los campos nuevos tienen valores por defecto para no romper a quienes ya
    construyen envelopes solo con ``source``, ``payload`` y ``stream_id``.
    """

    source: str
    payload: Mapping[str, Any]
    stream_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = SCHEMA_VERSION
    payload_version: int = DEFAULT_PAYLOAD_VERSION
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        # Sin correlación explícita, el evento inicia su propia cadena.
        if self.correlation_id is None:
            object.__setattr__(self, "correlation_id", self.event_id)
        validate_payload(self.payload)

    def derive(
        self,
        source: str,
        payload: Mapping[str, Any],
        *,
        suffix: str,
        stream_id: str | None = None,
        payload_version: int = DEFAULT_PAYLOAD_VERSION,
    ) -> MetadataEnvelope:
        """Crea un envelope hijo conservando la correlación.

        El ``event_id`` hijo es determinista (``<padre>/<suffix>``): reentregar
        el mismo mensaje de entrada produce el mismo identificador de salida,
        lo que permite deduplicar aguas abajo.
        """
        return MetadataEnvelope(
            source=source,
            payload=payload,
            stream_id=stream_id if stream_id is not None else self.stream_id,
            payload_version=payload_version,
            event_id=f"{self.event_id}/{suffix}",
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
        )


class MetadataHub(Protocol):
    """Interfaz que deberá adaptar el mecanismo de mensajería de AgentScope."""

    async def publish(self, topic: Topic, message: MetadataEnvelope) -> None:
        """Publica metadata en un canal tipado."""

    def subscribe(self, topic: Topic) -> AsyncIterator[MetadataEnvelope]:
        """Suscribe un consumidor a un canal tipado."""


# --- Versiones de payload por tópico -------------------------------------

_PAYLOAD_VERSIONS: dict[Topic, frozenset[int]] = {}


def supported_payload_versions(topic: Topic) -> frozenset[int]:
    """Versiones de payload aceptadas para un tópico."""
    return _PAYLOAD_VERSIONS.get(topic, frozenset({DEFAULT_PAYLOAD_VERSION}))


def register_payload_versions(topic: Topic, versions: Iterable[int]) -> None:
    """Declara las versiones de payload que acepta un tópico.

    Permite a un productor evolucionar su payload (p. ej. v2) sin tocar el
    transporte: primero se registra la versión y luego se publica.
    """
    _PAYLOAD_VERSIONS[topic] = frozenset(versions)


# --- Validación -----------------------------------------------------------


def parse_topic(value: str | Topic) -> Topic:
    """Convierte un texto en ``Topic`` o falla con un mensaje claro."""
    try:
        return Topic(value)
    except ValueError:
        permitidos = ", ".join(topic.value for topic in Topic)
        raise InvalidTopicError(
            f"Topic desconocido: {value!r}. Permitidos: {permitidos}"
        ) from None


_MAX_PAYLOAD_DEPTH = 32
_JSON_SCALARS = (str, int, float, bool, type(None))


def _looks_like_array(value: object) -> bool:
    """Detecta ndarray/tensores por duck typing, sin importar numpy ni torch."""
    return (
        hasattr(value, "__array_interface__")
        or hasattr(value, "__array__")
        or (hasattr(value, "shape") and hasattr(value, "dtype"))
    )


def _check_value(value: object, path: str, depth: int) -> None:
    if isinstance(value, _JSON_SCALARS):
        return
    if isinstance(value, bytes | bytearray | memoryview):
        raise InvalidEnvelopeError(
            f"Datos binarios en {path} ({type(value).__name__}): el bus solo "
            "transporta metadata, nunca frames de video"
        )
    if _looks_like_array(value):
        raise InvalidEnvelopeError(
            f"Array, tensor o escalar de array en {path} ({type(value).__name__}): el bus solo "
            "transporta metadata, nunca frames de video"
        )
    if depth >= _MAX_PAYLOAD_DEPTH:
        raise InvalidEnvelopeError(
            f"El payload supera la profundidad máxima ({_MAX_PAYLOAD_DEPTH}) en "
            f"{path}; ¿referencia circular?"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidEnvelopeError(
                    f"El payload no es serializable a JSON: clave no textual "
                    f"{key!r} en {path}"
                )
            _check_value(item, f"{path}.{key}", depth + 1)
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check_value(item, f"{path}[{index}]", depth + 1)
        return
    raise InvalidEnvelopeError(
        f"El payload no es serializable a JSON: tipo {type(value).__name__} "
        f"en {path} (solo str, int, float, bool, None, list, tuple y dict)"
    )


def validate_payload(payload: object) -> None:
    """Exige un Mapping de primitivas JSON: sin binarios, arrays ni objetos.

    Rechaza ``bytes``/``bytearray``/``memoryview``, arrays y tensores
    (ndarray, torch) detectados por duck typing, y cualquier tipo que no sea
    ``str``, ``int``, ``float``, ``bool``, ``None``, ``list``, ``tuple`` o
    ``dict`` con claves de texto. Así ningún frame puede llegar al bus.
    """
    if not isinstance(payload, Mapping):
        raise InvalidEnvelopeError(
            f"El payload debe ser un Mapping, no {type(payload).__name__}"
        )
    _check_value(payload, "payload", 0)


def _dump_payload(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    return json.dumps(payload, separators=(",", ":"))


def validate_envelope(topic: Topic | str, envelope: MetadataEnvelope) -> Topic:
    """Valida tópico, versiones y contenido del envelope.

    Devuelve el ``Topic`` normalizado. Lanza ``InvalidTopicError``,
    ``UnsupportedVersionError`` o ``InvalidEnvelopeError``.
    """
    parsed = parse_topic(topic)
    if envelope.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedVersionError(
            f"schema_version {envelope.schema_version} no soportada "
            f"(soportadas: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )
    versions = supported_payload_versions(parsed)
    if envelope.payload_version not in versions:
        raise UnsupportedVersionError(
            f"payload_version {envelope.payload_version} no soportada en "
            f"'{parsed.value}' (soportadas: {sorted(versions)})"
        )
    if not envelope.source:
        raise InvalidEnvelopeError("El campo source no puede estar vacío")
    if not envelope.event_id:
        raise InvalidEnvelopeError("El campo event_id no puede estar vacío")
    validate_payload(envelope.payload)
    return parsed


# --- Serialización --------------------------------------------------------


def envelope_to_dict(topic: Topic | str, envelope: MetadataEnvelope) -> dict[str, Any]:
    """Serializa un envelope validado a un dict JSON-compatible."""
    parsed = validate_envelope(topic, envelope)
    return {
        "topic": parsed.value,
        "source": envelope.source,
        "payload": json.loads(_dump_payload(envelope.payload)),
        "stream_id": envelope.stream_id,
        "created_at": envelope.created_at.isoformat(),
        "schema_version": envelope.schema_version,
        "payload_version": envelope.payload_version,
        "event_id": envelope.event_id,
        "correlation_id": envelope.correlation_id,
        "causation_id": envelope.causation_id,
    }


_REQUIRED_KEYS = ("topic", "source", "payload", "event_id", "schema_version")


def envelope_from_dict(data: Mapping[str, Any]) -> tuple[Topic, MetadataEnvelope]:
    """Reconstruye y valida un envelope a partir de ``envelope_to_dict``."""
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise InvalidEnvelopeError(f"Mensaje inválido: falta el campo '{key}'")
    topic = parse_topic(data["topic"])
    created_raw = data.get("created_at")
    envelope = MetadataEnvelope(
        source=data["source"],
        payload=data["payload"],
        stream_id=data.get("stream_id"),
        created_at=(
            datetime.fromisoformat(created_raw) if created_raw else datetime.now(UTC)
        ),
        schema_version=data["schema_version"],
        payload_version=data.get("payload_version", DEFAULT_PAYLOAD_VERSION),
        event_id=data["event_id"],
        correlation_id=data.get("correlation_id"),
        causation_id=data.get("causation_id"),
    )
    validate_envelope(topic, envelope)
    return topic, envelope


# --- Eventos de error -----------------------------------------------------


def build_error_envelope(
    source: str,
    error: BaseException,
    *,
    failed: MetadataEnvelope | None = None,
    failed_topic: Topic | None = None,
    stage: str = "handle",
    attempts: int = 1,
) -> MetadataEnvelope:
    """Construye el envelope publicado en ``Topic.ERRORS`` ante un fallo.

    Conserva la correlación del mensaje fallido para poder rastrear la cadena
    completa. El ``event_id`` es determinista por (mensaje fallido, origen).
    """
    payload = {
        "error_type": type(error).__name__,
        "message": str(error),
        "stage": stage,
        "failed_topic": failed_topic.value if failed_topic else None,
        "failed_event_id": failed.event_id if failed else None,
        "attempts": attempts,
    }
    if failed is None:
        return MetadataEnvelope(source=source, payload=payload)
    return failed.derive(source, payload, suffix=f"error/{source}")
