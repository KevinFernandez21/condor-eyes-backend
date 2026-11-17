"""Ayudas de prueba para WebSockets con el TestClient de Starlette.

Al salir de ``websocket_connect``, Starlette envía la desconexión y enseguida
cancela la tarea de la app (``cs.cancel``). Si el manejador aún no terminó su
limpieza (carga alta), la tarea se cancela a medias y ``fut.result`` lanza
``CancelledError``. ``ws_session`` se desconecta primero y espera, con tope, a
que el servidor termine la sesión antes de dejar que Starlette cancele.
"""

import contextlib
import time


def wait_until(condition, timeout: float = 5.0) -> bool:
    """Reintenta ``condition`` hasta que sea verdadera o venza ``timeout``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


@contextlib.contextmanager
def ws_session(client, url, **kwargs):
    with client.websocket_connect(url, **kwargs) as ws:
        try:
            yield ws
        finally:
            ws.close()  # el cliente se va; el servidor limpia su sesión
            assert wait_until(lambda: client.app.state.ws_clients_count() == 0), (
                "el servidor no cerró la sesión WebSocket a tiempo"
            )
