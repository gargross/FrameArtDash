"""Public-domain artwork from museum open-access APIs, used as the card background.

Sources (no API key needed):
- The Met:  https://collectionapi.metmuseum.org/public/collection/v1 (search → object → primaryImage)
- Cleveland Museum of Art: https://openaccess-api.clevelandart.org/api/artworks (cc0=1, images.print)

Only public-domain / CC0 works are used. A candidate is accepted when its image is landscape
(wide enough for a 16:9 screen) and at least ``min_width`` pixels wide; it is then cropped to
the output size and stored under ``<data>/gallery``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

import aiohttp
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"
CMA_API = "https://openaccess-api.clevelandart.org/api/artworks/"
SOURCES = ("met", "cleveland")
MIN_ASPECT = 1.25  # wider than 5:4, so a 16:9 crop keeps most of the picture
MAX_BYTES = 80 * 1024 * 1024
CANDIDATE_TTL = 24 * 3600
ATTEMPTS = 12
USER_AGENT = "FrameArtDash (+https://github.com/cutzenfriend/FrameArtDash)"


class GalleryError(RuntimeError):
    pass


@dataclass
class Artwork:
    source: str
    id: str
    title: str
    artist: str = ""
    date: str = ""
    url: str = ""  # museum page
    image_url: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def caption(self) -> str:
        who = ", ".join(p for p in (self.artist, self.date) if p)
        return f"{self.title} — {who}" if who else self.title

    def as_dict(self) -> dict:
        return asdict(self)


class Gallery:
    """Picks artworks. Candidate lists are cached per source and query for a day."""

    def __init__(
        self,
        directory: Path,
        met_api: str = MET_API,
        cma_api: str = CMA_API,
        rng: random.Random | None = None,
    ) -> None:
        self.dir = directory
        self.met_api = met_api.rstrip("/")
        self.cma_api = cma_api
        self.rng = rng or random.Random()
        self._cache: dict[tuple[str, str], tuple[float, list]] = {}

    # -- public ----------------------------------------------------------------

    async def pick(
        self,
        query: str,
        sources: tuple[str, ...],
        recent: list[str],
        min_width: int,
        size: tuple[int, int],
    ) -> tuple[Artwork, Path]:
        """Choose, download, crop and store a new artwork. Raises GalleryError if none fits."""
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
            tried: set[str] = set(recent)
            errors: list[str] = []
            for _ in range(ATTEMPTS):
                source = self.rng.choice([s for s in sources if s in SOURCES] or list(SOURCES))
                try:
                    art = await self._candidate(session, source, query, tried)
                except (aiohttp.ClientError, TimeoutError, ValueError, KeyError) as e:
                    errors.append(f"{source}: {e}")
                    continue
                if art is None:
                    continue
                tried.add(art.key)
                try:
                    data = await self._download(session, art.image_url)
                    jpeg = await asyncio.to_thread(_fit, data, min_width, size)
                except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
                    log.debug("Skipping %s: %s", art.key, e)
                    continue
                path = self._store(art, jpeg)
                log.info("Gallery: %s (%s)", art.caption, art.key)
                return art, path
            raise GalleryError(
                "no suitable artwork found" + (f" ({'; '.join(errors[-3:])})" if errors else "")
            )

    async def search(self, source: str, query: str, page: int = 0, per_page: int = 24) -> dict:
        """One page of public-domain paintings for the gallery browser."""
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
            if source == "cleveland":
                params = {
                    "q": query or "landscape",
                    "type": "Painting",
                    "cc0": "1",
                    "has_image": "1",
                    "limit": str(per_page),
                    "skip": str(page * per_page),
                }
                data = await _json(session, f"{self.cma_api}?{urlencode(params)}")
                items = [_cma_item(a) for a in data.get("data") or []]
                total = int((data.get("info") or {}).get("total") or len(items))
                return {
                    "items": [i for i in items if i],
                    "total": total,
                    "page": page,
                    "more": (page + 1) * per_page < total,
                }
            if source != "met":
                raise ValueError(f"unknown source {source!r}")
            ids = await self._cached(("met", query), lambda: self._met_ids(session, query))
            chunk = ids[page * per_page : (page + 1) * per_page]
            sem = asyncio.Semaphore(6)

            async def one(object_id):
                async with sem:
                    try:
                        o = await _json(session, f"{self.met_api}/objects/{object_id}")
                    except (aiohttp.ClientError, TimeoutError, ValueError):
                        return None
                return _met_item(object_id, o)

            items = await asyncio.gather(*(one(i) for i in chunk))
            return {
                "items": [i for i in items if i],
                "total": len(ids),
                "page": page,
                "more": (page + 1) * per_page < len(ids),
            }

    async def fetch(self, key: str, size: tuple[int, int], min_width: int = 1280) -> tuple[Artwork, Path]:
        """Download one specific artwork (``met:123`` / ``cleveland:456``) chosen by hand."""
        source, _, ident = key.partition(":")
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}) as session:
            if source == "met":
                art = await self._met_object(session, ident)
            elif source == "cleveland":
                data = await _json(session, f"{self.cma_api.rstrip('/')}/{ident}")
                item = _cma_item(data.get("data") or {})
                art = (
                    Artwork(
                        **{
                            k: item[k]
                            for k in ("source", "id", "title", "artist", "date", "url", "image_url")
                        }
                    )
                    if item
                    else None
                )
            else:
                raise GalleryError(f"unknown artwork {key!r}")
            if art is None:
                raise GalleryError(f"{key} is not a public-domain work with an image")
            try:
                data = await self._download(session, art.image_url)
                jpeg = await asyncio.to_thread(_fit, data, min_width, size, 0)
            except (aiohttp.ClientError, TimeoutError, OSError, ValueError) as e:
                raise GalleryError(f"{key}: {e}") from e
        return art, self._store(art, jpeg)

    def current_path(self, art: dict | None) -> Path | None:
        if not art or not art.get("file"):
            return None
        path = self.dir / art["file"]
        return path if path.exists() else None

    # -- sources -----------------------------------------------------------------

    async def _candidate(self, session, source: str, query: str, tried: set[str]) -> Artwork | None:
        if source == "met":
            ids = await self._cached(("met", query), lambda: self._met_ids(session, query))
            ids = [i for i in ids if f"met:{i}" not in tried]
            if not ids:
                return None
            object_id = self.rng.choice(ids)
            tried.add(f"met:{object_id}")
            return await self._met_object(session, object_id)
        arts = await self._cached(("cleveland", query), lambda: self._cma_artworks(session, query))
        arts = [a for a in arts if a.key not in tried]
        return self.rng.choice(arts) if arts else None

    async def _cached(self, key, fetch):
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < CANDIDATE_TTL:
            return hit[1]
        items = await fetch()
        self._cache[key] = (time.monotonic(), items)
        return items

    async def _met_ids(self, session, query: str) -> list[str]:
        params = {"hasImages": "true", "medium": "Paintings", "q": query or "landscape"}
        data = await _json(session, f"{self.met_api}/search?{urlencode(params)}")
        return [str(i) for i in (data.get("objectIDs") or [])]

    async def _met_object(self, session, object_id: str) -> Artwork | None:
        o = await _json(session, f"{self.met_api}/objects/{object_id}")
        if not o.get("isPublicDomain") or not o.get("primaryImage"):
            return None
        return Artwork(
            source="met",
            id=str(object_id),
            title=(o.get("title") or "Untitled").strip(),
            artist=(o.get("artistDisplayName") or "").strip(),
            date=(o.get("objectDate") or "").strip(),
            url=o.get("objectURL") or "",
            image_url=o["primaryImage"],
        )

    async def _cma_artworks(self, session, query: str) -> list[Artwork]:
        params = {"q": query or "landscape", "type": "Painting", "cc0": "1", "has_image": "1", "limit": "200"}
        data = await _json(session, f"{self.cma_api}?{urlencode(params)}")
        arts = []
        for a in data.get("data") or []:
            item = _cma_item(a)
            if not item or item["landscape"] is False:
                continue  # portrait or square: skip without downloading
            arts.append(
                Artwork(
                    **{k: item[k] for k in ("source", "id", "title", "artist", "date", "url", "image_url")}
                )
            )
        return arts

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


async def _json(session, url: str) -> dict:
    async with session.get(url, headers={"Accept": "application/json"}) as resp:
        resp.raise_for_status()
        text = await resp.text()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("unexpected response")
    return data


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


def _met_item(object_id, o: dict) -> dict | None:
    if not o.get("isPublicDomain") or not o.get("primaryImage"):
        return None
    return {
        "key": f"met:{object_id}",
        "source": "met",
        "id": str(object_id),
        "title": (o.get("title") or "Untitled").strip(),
        "artist": (o.get("artistDisplayName") or "").strip(),
        "date": (o.get("objectDate") or "").strip(),
        "url": o.get("objectURL") or "",
        "image_url": o["primaryImage"],
        "thumb": o.get("primaryImageSmall") or o["primaryImage"],
        "width": None,
        "height": None,
        "landscape": None,
    }


def _cma_item(a: dict) -> dict | None:
    images = a.get("images") or {}
    img = images.get("print") or {}
    if a.get("share_license_status") not in (None, "CC0") or not img.get("url"):
        return None
    try:
        width, height = int(img.get("width") or 0), int(img.get("height") or 0)
    except ValueError:
        width = height = 0
    creators = a.get("creators") or []
    artist = (creators[0].get("description") or "").split("(")[0].strip() if creators else ""
    return {
        "key": f"cleveland:{a.get('id')}",
        "source": "cleveland",
        "id": str(a.get("id")),
        "title": (a.get("title") or "Untitled").strip(),
        "artist": artist,
        "date": (a.get("creation_date") or "").strip(),
        "url": a.get("url") or "",
        "image_url": img["url"],
        "thumb": (images.get("web") or {}).get("url") or img["url"],
        "width": width or None,
        "height": height or None,
        "landscape": (width >= height * MIN_ASPECT) if width and height else None,
    }


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
