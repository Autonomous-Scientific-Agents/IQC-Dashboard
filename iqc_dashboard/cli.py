"""Command-line interface for IQC Dashboard."""

import argparse
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


DEFAULT_PORT = 8501
PORT_SEARCH_LIMIT = 100


def _is_port_available(host: str, port: int) -> bool:
    """Return True when Streamlit can bind to host:port.

    Note: this is a best-effort check. Between the moment we release the probe
    socket here and the moment Streamlit rebinds in the child process another
    process can grab the port. Streamlit itself will fail loudly if that
    happens, so we accept the race rather than plumbing an inherited fd.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Match how Streamlit itself binds so a socket lingering in
            # TIME_WAIT does not produce a false "in use" result on macOS.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError:
        return False
    return True


def _find_available_port(host: str, start_port: int = DEFAULT_PORT) -> int:
    """Find the first available port at or above start_port."""
    for port in range(start_port, start_port + PORT_SEARCH_LIMIT):
        if _is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No available port found between {start_port} and "
        f"{start_port + PORT_SEARCH_LIMIT - 1}."
    )


def _valid_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc

    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _parse_args(argv: Optional[Sequence[str]]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        prog="iqc-dashboard",
        description="Run the IQC Dashboard Streamlit app.",
    )
    parser.add_argument(
        "--port",
        "--server.port",
        dest="port",
        type=_valid_port,
        default=None,
        help="Port to run Streamlit on. Defaults to the first free port at or above 8501.",
    )
    parser.add_argument(
        "--host",
        "--server.address",
        dest="host",
        default="localhost",
        help="Host interface for Streamlit to bind to. Defaults to localhost.",
    )
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the Streamlit app."""
    args, streamlit_args = _parse_args(argv)

    package_dir = Path(__file__).parent
    app_file = package_dir / "app.py"

    try:
        port = args.port
        if port is None:
            port = _find_available_port(args.host)
            if port != DEFAULT_PORT:
                print(
                    f"Port {DEFAULT_PORT} is in use; starting IQC Dashboard on port {port}.",
                    file=sys.stderr,
                )
        elif not _is_port_available(args.host, port):
            print(f"Port {port} is not available on {args.host}.", file=sys.stderr)
            sys.exit(1)

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            f"--server.address={args.host}",
            f"--server.port={port}",
            f"--browser.serverPort={port}",
        ]
        command.extend(streamlit_args)

        subprocess.run(command, check=True)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
