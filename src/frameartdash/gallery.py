"""Public-domain artwork from museum open-access collections, used as the card background.

The museums themselves live in :mod:`frameartdash.museums`. This module picks artworks for the
rotation, serves the gallery browser's searches, downloads and crops the chosen image, and keeps
the user's collection and schedule (:class:`Library`).

Search terms: a comma separates alternatives (OR). Each rotation picks one alternative at random;
the browser searches all of them. Words within one alternative go to the museum's own search
(the National Gallery of Art, searched locally, requires every word).

Subjects to leave out (GALLERY_EXCLUDE) are matched as word prefixes against the title and, where
the museum provides them, its subject tags. Manual "Show now" picks are not filtered.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import time
from pathlib import Path

import aiohttp
from PIL import Image, ImageOps

from frameartdash.museums import (
    AIC,
    LABELS,
    MIN_ASPECT,
    NGA,
    SOURCES,
    Artwork,
    Cleveland,
    Met,
    Rijks,
)

__all__ = ["Artwork", "Gallery", "GalleryError", "Library", "LABELS", "SOURCES", "excluded", "split_terms"]

log = logging.getLogger(__name__)

MAX_BYTES = 80 * 1024 * 1024
CANDIDATE_TTL = 24 * 3600
ATTEMPTS = 14
USER_AGENT = "FrameArtDash (+https://github.com/cutzenfriend/FrameArtDash)"


class GalleryError(RuntimeError):
    pass


def split_terms(query: str) -> list[str]:
    """``'seascape, winter harbor'`` → ``['seascape', 'winter harbor']``."""
    return [t.strip() for t in (query or "").split(",") if t.strip()]


def excluded(title: str, tags, words) -> str | None:
    """The first exclusion word found (as a word prefix) in the title or tags, or None."""
    text = " | ".join([title or "", *(tags or [])])
    for w in words or ():
        if w and re.search(r"(?<![\w])" + re.escape(w), text, re.IGNORECASE):
            return w
    return None


class Gallery:
    """Picks artworks. Candidate lists are cached per source and search term for a day."""

    def __init__(
        self,
        directory: Path,
        met_api: str | None = None,
        cma_api: str | None = None,
        aic_api: str | None = None,
        aic_iiif: str | None = None,
        nga_data: str | None = None,
        rijks_api: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.dir = directory
        self.rng = rng or random.Random()
        self.sources = {
            "met": Met(met_api) if met_api else Met(),
            "cleveland": Cleveland(cma_api) if cma_api else Cleveland(),
            "aic": AIC(*(a for a in (aic_api, aic_iiif) if a)) if aic_api else AIC(),
            "nga": NGA(directory, nga_data) if nga_data else NGA(directory),
            "rijks": Rijks(rijks_api) if rijks_api else Rijks(),
        }
        self._cache: dict[tuple[str, str], tuple[float, list]] = {}

    def _session(self, total: int):
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=total),
            headers={"User-Agent": USER_AGENT, "AIC-User-Agent": USER_AGENT},
        )

    # -- rotation --------------------------------------------------------------

    async def pick(
        self,
        query: str,
        sources: tuple[str, ...],
        recent: list[str],
        min_width: int,
        size: tuple[int, int],
        exclude: tuple[str, ...] = (),
    ) -> tuple[Artwork, Path]:
        """Choose, download, crop and store a new artwork. Raises GalleryError if none fits."""
        terms = split_terms(query) or ["landscape"]
        usable = [s for s in sources if s in self.sources] or list(SOURCES)
        async with self._session(180) as session:
            tried: set[str] = set(recent)
            errors: list[str] = []
            for _ in range(ATTEMPTS):
                source, term = self.rng.choice(usable), self.rng.choice(terms)
                src = self.sources[source]
                try:
                    items = await self._cached((source, term), lambda s=src, t=term: s.candidates(session, t))
                    fresh = [i for i in items if i["key"] not in tried]
                    if not fresh:
                        continue
                    it = self.rng.choice(fresh)
                    tried.add(it["key"])
                    if it.get("title") and excluded(it["title"], it.get("tags"), exclude):
                        continue
                    art = await src.resolve(session, it)
                except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, OSError) as e:
                    errors.append(f"{source}: {e or type(e).__name__}")
                    continue
                if art is None:
                    continue
                if hit := excluded(art.title, art.tags, exclude):
                    log.debug("Skipping %s (%s): subject '%s' is excluded", art.key, art.title, hit)
                    continue
                try:
                    jpeg = await self._image(session, art, min_width, size, MIN_ASPECT)
                except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
                    log.debug("Skipping %s: %s", art.key, e)
                    continue
                path = self._store(art, jpeg)
                log.info("Gallery: %s (%s)", art.caption, art.key)
                return art, path
            raise GalleryError(
                "no suitable artwork found" + (f" ({'; '.join(errors[-3:])})" if errors else "")
            )

    async def _image(self, session, art: Artwork, min_width, size, min_aspect) -> bytes:
        last: Exception | None = None
        for url in [art.image_url, *art.fallback_urls]:
            try:
                data = await self._download(session, url)
                return await asyncio.to_thread(_fit, data, min_width, size, min_aspect)
            except aiohttp.ClientResponseError as e:
                last = e  # e.g. a size the image server refuses: try the next one
            except ValueError as e:
                last = e
                if "too small" not in str(e):
                    break
        raise last or ValueError("no image")

    # -- browser ---------------------------------------------------------------

    async def search(
        self, source: str, query: str, page: int = 0, per_page: int = 24, exclude: tuple[str, ...] = ()
    ) -> dict:
        """One page of public-domain paintings for the gallery browser (all comma alternatives)."""
        if source not in self.sources:
            raise ValueError(f"unknown source {source!r}")
        src = self.sources[source]
        terms = split_terms(query) or ["landscape"]
        async with self._session(120) as session:
            pages = await asyncio.gather(*(src.search(session, t, page, per_page) for t in terms))
        items, seen, hidden = [], set(), 0
        for p in pages:
            for it in p["items"]:
                if it["key"] in seen:
                    continue
                seen.add(it["key"])
                if excluded(it["title"], it.get("tags"), exclude):
                    hidden += 1
                    continue
                items.append({k: v for k, v in it.items() if k != "medium"})
        return {
            "items": items,
            "total": sum(p["total"] for p in pages),
            "page": page,
            "more": any(p["more"] for p in pages),
            "hidden": hidden,
        }

    async def fetch(self, key: str, size: tuple[int, int], min_width: int = 1280) -> tuple[Artwork, Path]:
        """Download one specific artwork (``met:123``, ``nga:46``…) chosen by hand."""
        source, _, ident = key.partition(":")
        if source not in self.sources or not ident:
            raise GalleryError(f"unknown artwork {key!r}")
        async with self._session(240) as session:
            try:
                art = await self.sources[source].by_id(session, ident)
            except (aiohttp.ClientError, TimeoutError, ValueError, KeyError) as e:
                raise GalleryError(f"{key}: {e or type(e).__name__}") from e
            if art is None:
                raise GalleryError(f"{key} is not a public-domain work with an image")
            try:
                jpeg = await self._image(session, art, min_width, size, 0)
            except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
                raise GalleryError(f"{key}: {e}") from e
        return art, self._store(art, jpeg)

    def current_path(self, art: dict | None) -> Path | None:
        if not art or not art.get("file"):
            return None
        path = self.dir / art["file"]
        return path if path.exists() else None

    # -- helpers ---------------------------------------------------------------

    async def _cached(self, key, fetch):
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CANDIDATE_TTL:
            return hit[1]
        items = await fetch()
        self._cache[key] = (time.monotonic(), items)
        return items

    async def _download(self, session, url: str) -> bytes:
        async with session.get(url) as resp:
            resp.raise_for_status()
            if resp.content_length and resp.content_length > MAX_BYTES:
                raise ValueError(f"image too large ({resp.content_length} bytes)")
            chunks, total = [], 0
            async for chunk in resp.content.iter_chunked(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("image too large")
                chunks.append(chunk)
        return b"".join(chunks)

    def _store(self, art: Artwork, jpeg: bytes) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        name = f"{art.source}-{art.id}.jpg"
        tmp = self.dir / (name + ".tmp")
        tmp.write_bytes(jpeg)
        target = self.dir / name
        tmp.replace(target)
        for old in self.dir.glob("*.jpg"):
            if old != target:
                old.unlink(missing_ok=True)
        return target


def _fit(data: bytes, min_width: int, size: tuple[int, int], min_aspect: float = MIN_ASPECT) -> bytes:
    """Validate and crop to the output size. Raises ValueError for unsuitable images."""
    Image.MAX_IMAGE_PIXELS = 400_000_000  # museum scans are large but trusted sources
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        if width < min_width:
            raise ValueError(f"too small ({width}px wide)")
        if min_aspect and width < height * min_aspect:
            raise ValueError(f"not landscape ({width}x{height})")
        fitted = ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)
    out = io.BytesIO()
    fitted.save(out, "JPEG", quality=92)
    return out.getvalue()


ITEM_FIELDS = ("key", "source", "id", "title", "artist", "date", "url", "thumb")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


class Library:
    """The user's saved collection and rotation schedule, stored next to the artwork files."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _read(self, name: str, default):
        try:
            return json.loads((self.dir / name).read_text())
        except FileNotFoundError:
            return default
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Could not read %s (%s)", name, e)
            return default

    def _write(self, name: str, data) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.dir / name)

    # collection
    def collection(self) -> list[dict]:
        return list(self._read("collection.json", []))

    def save_collection(self, items: list) -> list[dict]:
        clean, seen = [], set()
        for raw in items or []:
            if not isinstance(raw, dict):
                raise ValueError("collection items must be objects")
            item = {k: str(raw.get(k) or "") for k in ITEM_FIELDS}
            source, _, ident = item["key"].partition(":")
            if source not in SOURCES or not ident:
                raise ValueError(f"invalid artwork key {item['key']!r}")
            item["source"], item["id"] = source, ident
            if item["key"] not in seen:
                seen.add(item["key"])
                clean.append(item)
        self._write("collection.json", clean)
        return clean

    # schedule
    def schedule(self) -> dict:
        return self._read("schedule.json", {"rules": []})

    def save_schedule(self, data: dict) -> dict:
        from frameartdash.config import parse_active_hours

        rules = []
        for raw in (data or {}).get("rules") or []:
            days = [d for d in (raw.get("days") or list(WEEKDAYS)) if d in WEEKDAYS]
            start, end = str(raw.get("start") or ""), str(raw.get("end") or "")
            parse_active_hours(f"{start}-{end}")  # raises ConfigError on bad times
            source = raw.get("source") if raw.get("source") in ("search", "collection") else "search"
            rules.append(
                {
                    "days": days or list(WEEKDAYS),
                    "start": start,
                    "end": end,
                    "source": source,
                    "query": str(raw.get("query") or "").strip(),
                }
            )
        saved = {"rules": rules}
        self._write("schedule.json", saved)
        return saved

    def active_rule(self, now) -> dict | None:
        """The first rule whose days and time window contain ``now`` (local time)."""
        from frameartdash.config import in_window, parse_active_hours

        day = WEEKDAYS[now.weekday()]
        for rule in self.schedule().get("rules") or []:
            try:
                window = parse_active_hours(f"{rule['start']}-{rule['end']}")
            except Exception:  # noqa: BLE001 - a broken rule must not stop rotation
                continue
            if day in rule.get("days", WEEKDAYS) and in_window(window, now.time()):
                return rule
        return None
