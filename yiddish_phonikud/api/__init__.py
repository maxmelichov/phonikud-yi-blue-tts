"""HTTP layer: DTOs plus the two routers mounted by `app.create_app()`."""

from __future__ import annotations

from . import dto, routes_ui, routes_v1, routes_lexicon

__all__ = ["dto", "routes_lexicon", "routes_ui", "routes_v1"]
