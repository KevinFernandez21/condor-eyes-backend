"""CLI del receptor: RSSI en vivo y dentro/fuera del área del tag enrolado."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from .config import load_site_map
from .presence import PresenceTracker
from .receiver import ReceiverEvent, TagReceiver, run_receiver
from .scanner import BleakAdvertisementSource

DEFAULT_SITE_MAP = Path(__file__).resolve().parents[2] / "configs" / "site_map.toml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Receptor BLE del tag XIAO ESP32-C6 (issue #26)."
    )
    p.add_argument("--tag", required=True, help="ID del tag enrolado (12 dígitos hex)")
    p.add_argument("--site-map", type=Path, default=DEFAULT_SITE_MAP)
    p.add_argument(
        "--zone", default=None, help="zona del site map (por defecto la única)"
    )
    p.add_argument(
        "--enter-dbm", type=float, default=None, help="umbral de entrada (dBm)"
    )
    p.add_argument("--hysteresis-db", type=float, default=None, help="histéresis (dB)")
    p.add_argument("--window", type=int, default=5, help="muestras del suavizado")
    p.add_argument("--lost-after-s", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=None, help="segundos a escanear")
    p.add_argument(
        "--jsonl", action="store_true", help="imprime observaciones payload v1"
    )
    return p


def format_event(event: ReceiverEvent) -> str:
    st = event.state
    stamp = datetime.fromtimestamp(st.ts).astimezone().strftime("%H:%M:%S.%f")[:-3]
    if event.observation is None:
        return (
            f"{stamp}  sin señal  {st.status:<7} conf={st.confidence:.2f} ({st.reason})"
        )
    obs = event.observation
    bat = f"{obs['bat']}%" if "bat" in obs else "?"
    return (
        f"{stamp}  rssi={obs['rssi']:>4} dBm  media={st.smoothed_dbm:6.1f}  "
        f"{st.status:<7} conf={st.confidence:.2f}  seq={obs['seq']}  bat={bat}"
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
    tracker = PresenceTracker(
        enter_dbm=enter,
        hysteresis_db=hyst,
        window=args.window,
        lost_after_s=args.lost_after_s,
    )
    receiver = TagReceiver(enrolled=[args.tag], node_id=zone.receiver, tracker=tracker)
    print(
        f"Zona {zone.id!r} | receptor {zone.receiver} | entra >= {enter} dBm, "
        f"sale < {enter - hyst} dBm | calibrado: {'sí' if zone.calibrated else 'NO'}",
        flush=True,
    )
    last_status: str | None = None

    def sink(event: ReceiverEvent) -> None:
        nonlocal last_status
        if args.jsonl:
            if event.observation is not None:
                print(json.dumps(event.observation, separators=(",", ":")), flush=True)
            return
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
    print(f"Resumen: {receiver.stats}", flush=True)
