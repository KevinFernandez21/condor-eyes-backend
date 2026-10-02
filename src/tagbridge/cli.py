"""CLI del receptor: RSSI en vivo y dentro/fuera del área del tag enrolado."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from location import (
    InMemoryPersonnelRepository,
    LocationService,
    PersonnelRecord,
    Pseudonymizer,
    load_config,
)

from .config import load_site_map
from .presence import PresenceView
from .receiver import ReceiverEvent, TagReceiver, run_receiver
from .scanner import BleakAdvertisementSource

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receptor BLE del tag XIAO ESP32-C6 (issue #26)."
    )
    p.add_argument("--tag", required=True, help="ID del tag enrolado (12 dígitos hex)")
    p.add_argument("--site-map", type=Path, default=CONFIGS / "site_map.toml")
    p.add_argument("--location-config", type=Path, default=CONFIGS / "location.toml")
    p.add_argument(
        "--zone", default=None, help="zona del site map (por defecto la única)"
    )
    p.add_argument(
        "--enter-dbm", type=float, default=None, help="umbral de entrada (dBm)"
    )
    p.add_argument("--hysteresis-db", type=float, default=None, help="histéresis (dB)")
    p.add_argument("--duration", type=float, default=None, help="segundos a escanear")
    p.add_argument(
        "--jsonl", action="store_true", help="imprime observaciones payload v1"
    )
    return p


def format_event(event: ReceiverEvent) -> str:
    st = event.state
    if event.observation is None:
        return f"sin señal  {st.status:<7} conf={st.confidence:.2f} ({st.reason})"
    stamp = datetime.fromtimestamp(event.observation["ts"] / 1000).astimezone()
    obs = event.observation
    bat = f"{obs['bat']}%" if "bat" in obs else "?"
    return (
        f"{stamp:%H:%M:%S.%f}"[:-3]
        + f"  rssi={obs['rssi']:>4} dBm  media={st.smoothed_dbm:6.1f}  "
        f"zona={st.zone_id} {st.status:<7} conf={st.confidence:.2f}  "
        f"seq={obs['seq']}  bat={bat}"
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    site = load_site_map(args.site_map)
    if args.zone is None:
        if len(site.zones) != 1:
            raise SystemExit("Hay varias zonas: indica --zone.")
        zone = next(iter(site.zones.values()))
    else:
        zone = site.zone(args.zone)
    enter = zone.enter_dbm if args.enter_dbm is None else args.enter_dbm
    hyst = zone.hysteresis_db if args.hysteresis_db is None else args.hysteresis_db

    # El tag se enrola por el repositorio; los logs solo muestran seudónimos.
    key = os.environ.get("CONDOR_PSEUDONYM_KEY")
    pseudo = Pseudonymizer(key.encode()) if key else Pseudonymizer.random()
    repo = InMemoryPersonnelRepository(pseudo)
    repo.enroll(
        args.tag, PersonnelRecord("tag-owner", "Tag de prueba", frozenset({zone.id}))
    )
    service = LocationService(load_config(args.location_config), repo, pseudo)
    receiver = TagReceiver(
        service=service,
        enrolled=[args.tag],
        node_id=zone.receiver,
        view=PresenceView(enter_dbm=enter, hysteresis_db=hyst),
    )
    print(
        f"Zona {zone.id!r} | receptor {zone.receiver} | entra >= {enter} dBm, "
        f"sale < {enter - hyst} dBm | calibrado: {'sí' if zone.calibrated else 'NO'}",
        flush=True,
    )
    last_status: str | None = None

    def sink(event: ReceiverEvent) -> None:
        nonlocal last_status
        accepted = event.ingest is not None and event.ingest.accepted
        if args.jsonl:
            if accepted:
                print(json.dumps(event.observation, separators=(",", ":")), flush=True)
            return
        if event.observation is not None and not accepted:
            return  # rechazos: solo en el resumen, por motivo
        if event.observation is None and event.state.status == last_status:
            return  # tick sin cambios: no ensucia la salida
        last_status = event.state.status
        print(format_event(event), flush=True)

    try:
        asyncio.run(
            run_receiver(
                BleakAdvertisementSource(), receiver, sink, duration_s=args.duration
            )
        )
    except KeyboardInterrupt:
        pass
    rejected = {r.value: n for r, n in service.stats.items()}
    print(
        f"Resumen: aceptadas={service.accepted_count} rechazos={rejected} "
        f"ignoradas(otros dispositivos)={receiver.ignored}",
        flush=True,
    )
