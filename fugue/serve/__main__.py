from __future__ import annotations

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised in the image
        raise RuntimeError("Uvicorn is not installed; install fugue[serve]") from exc

    from fugue.serve.app import create_app

    uvicorn.run(
        create_app(),
        host=os.environ.get("FUGUE_SERVE_HOST", "0.0.0.0"),
        port=int(os.environ.get("FUGUE_SERVE_PORT", "8000")),
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
