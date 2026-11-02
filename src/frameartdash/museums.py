"""Museum open-access sources for the gallery.

Every source offers the same four operations on top of an aiohttp session:

- ``candidates(query)``: artworks that may be shown for a search term (cached by the caller),
  as item dicts; ``image_url`` may still be missing (``resolve`` fills it in).
- ``resolve(item)``: the complete :class:`Artwork`, or None if it is not usable.
- ``search(query, page, per_page)``: one page of items for the gallery browser.
- ``by_id(ident)``: one artwork chosen by hand.

Item dicts carry: key, source, id, title, artist, date, url, image_url, thumb, width, height,
landscape, tags. Only public-domain / CC0 works are returned.

Sources:
- met        The Met — collectionapi.metmuseum.org (search → object; isPublicDomain)
- cleveland  Cleveland Museum of Art — openaccess-api.clevelandart.org (cc0=1)
- aic        Art Institute of Chicago — api.artic.edu (is_public_domain) + IIIF images
- nga        National Gallery of Art — daily CSV open data on GitHub (openaccess=1) + IIIF
- rijks      Rijksmuseum — data.rijksmuseum.nl Linked Art search + iiif.micr.io
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

log = logging.getLogger(__name__)

MIN_ASPECT = 1.25  # wider than 5:4, so a 16:9 crop keeps most of the picture
LABELS = {
    "met": "The Met",
    "cleveland": "Cleveland Museum of Art",
    "aic": "Art Institute of Chicago",
    "nga": "National Gallery of Art",
    "rijks": "Rijksmuseum",
}
SOURCES = tuple(LABELS)


@dataclass
class Artwork:
    source: str
    id: str
    title: str
    artist: str = ""
    date: str = ""
    url: str = ""  # museum page
    image_url: str = ""
    fallback_urls: list[str] = field(default_factory=list)  # smaller sizes if the first is refused
    tags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.id}"

    @property
    def caption(self) -> str:
        who = ", ".join(p for p in (self.artist, self.date) if p)
        return f"{self.title} — {who}" if who else self.title

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("fallback_urls", None)
        return d


def item(
    source, ident, title, artist="", date="", url="", image_url="", thumb="", width=None, height=None, tags=()
):
    width = int(width) if width else None
    height = int(height) if height else None
    return {
        "key": f"{source}:{ident}",
        "source": source,
        "id": str(ident),
        "title": (title or "Untitled").strip(),
        "artist": (artist or "").strip(),
        "date": (date or "").strip(),
        "url": url or "",
        "image_url": image_url or "",
        "thumb": thumb or image_url or "",
        "width": width,
        "height": height,
        "landscape": (width >= height * MIN_ASPECT) if width and height else None,
        "tags": [t for t in tags if t],
    }


def artwork_from(it: dict, fallback_urls=()) -> Artwork:
    return Artwork(
        source=it["source"],
        id=it["id"],
        title=it["title"],
        artist=it["artist"],
        date=it["date"],
        url=it["url"],
        image_url=it["image_url"],
        fallback_urls=list(fallback_urls),
        tags=list(it.get("tags") or []),
    )


async def get_json(session, url: str, **kw) -> dict:
    async with session.get(url, headers={"Accept": "application/json"}, **kw) as resp:
        resp.raise_for_status()
        data = json.loads(await resp.text())
    if not isinstance(data, dict):
        raise ValueError("unexpected response")
    return data


def _page(items: list, page: int, per_page: int, total: int | None = None) -> dict:
    total = len(items) if total is None else total
    return {"items": items, "total": total, "page": page, "more": (page + 1) * per_page < total}


# -- The Met --------------------------------------------------------------------


class Met:
    name = "met"

    def __init__(self, api: str = "https://collectionapi.metmuseum.org/public/collection/v1") -> None:
        self.api = api.rstrip("/")

    async def _ids(self, session, query):
        params = {"hasImages": "true", "medium": "Paintings", "q": query or "landscape"}
        data = await get_json(session, f"{self.api}/search?{urlencode(params)}")
        return [str(i) for i in (data.get("objectIDs") or [])]

    def _item(self, object_id, o):
        if not o.get("isPublicDomain") or not o.get("primaryImage"):
            return None
        tags = [t.get("term", "") for t in (o.get("tags") or []) if isinstance(t, dict)]
        return item(
            "met",
            object_id,
            o.get("title"),
            o.get("artistDisplayName"),
            o.get("objectDate"),
            o.get("objectURL"),
            o["primaryImage"],
            o.get("primaryImageSmall") or o["primaryImage"],
            tags=tags,
        )

    async def candidates(self, session, query):
        return [{"key": f"met:{i}", "source": "met", "id": i} for i in await self._ids(session, query)]

    async def resolve(self, session, it):
        return await self.by_id(session, it["id"])

    async def by_id(self, session, ident):
        it = self._item(ident, await get_json(session, f"{self.api}/objects/{ident}"))
        return artwork_from(it) if it else None

    async def search(self, session, query, page, per_page):
        ids = await self._ids(session, query)
        chunk = ids[page * per_page : (page + 1) * per_page]
        sem = asyncio.Semaphore(6)

        async def one(object_id):
            async with sem:
                try:
                    return self._item(object_id, await get_json(session, f"{self.api}/objects/{object_id}"))
                except (aiohttp.ClientError, TimeoutError, ValueError):
                    return None

        items = await asyncio.gather(*(one(i) for i in chunk))
        return _page([i for i in items if i], page, per_page, len(ids))


# -- Cleveland ------------------------------------------------------------------


class Cleveland:
    name = "cleveland"

    def __init__(self, api: str = "https://openaccess-api.clevelandart.org/api/artworks/") -> None:
        self.api = api

    def _item(self, a):
        images = a.get("images") or {}
        img = images.get("print") or {}
        if a.get("share_license_status") not in (None, "CC0") or not img.get("url"):
            return None
        creators = a.get("creators") or []
        artist = (creators[0].get("description") or "").split("(")[0].strip() if creators else ""
        try:
            width, height = int(img.get("width") or 0), int(img.get("height") or 0)
        except ValueError:
            width = height = 0
        return item(
            "cleveland",
            a.get("id"),
            a.get("title"),
            artist,
            a.get("creation_date"),
            a.get("url"),
            img["url"],
            (images.get("web") or {}).get("url") or img["url"],
            width,
            height,
        )

    async def _query(self, session, query, limit, skip=0):
        params = {
            "q": query or "landscape",
            "type": "Painting",
            "cc0": "1",
            "has_image": "1",
            "limit": str(limit),
            "skip": str(skip),
        }
        return await get_json(session, f"{self.api}?{urlencode(params)}")

    async def candidates(self, session, query):
        data = await self._query(session, query, 200)
        return [
            i for i in (self._item(a) for a in data.get("data") or []) if i and i["landscape"] is not False
        ]

    async def resolve(self, session, it):
        return artwork_from(it)

    async def by_id(self, session, ident):
        it = self._item((await get_json(session, f"{self.api.rstrip('/')}/{ident}")).get("data") or {})
        return artwork_from(it) if it else None

    async def search(self, session, query, page, per_page):
        data = await self._query(session, query, per_page, page * per_page)
        items = [i for i in (self._item(a) for a in data.get("data") or []) if i]
        total = int((data.get("info") or {}).get("total") or len(items))
        return _page(items, page, per_page, total)


# -- Art Institute of Chicago -----------------------------------------------------

AIC_FIELDS = (
    "id,title,artist_title,artist_display,date_display,image_id,thumbnail,is_public_domain,"
    "artwork_type_title,subject_titles,term_titles"
)


class AIC:
    name = "aic"

    def __init__(
        self, api: str = "https://api.artic.edu/api/v1", iiif: str = "https://www.artic.edu/iiif/2"
    ) -> None:
        self.api = api.rstrip("/")
        self.iiif = iiif.rstrip("/")

    def _item(self, a):
        if not a.get("is_public_domain") or not a.get("image_id"):
            return None
        if (a.get("artwork_type_title") or "Painting") != "Painting":
            return None
        thumb = a.get("thumbnail") or {}
        base = f"{self.iiif}/{a['image_id']}"
        artist = a.get("artist_title") or (a.get("artist_display") or "").split("\n")[0]
        return item(
            "aic",
            a["id"],
            a.get("title"),
            artist,
            a.get("date_display"),
            f"https://www.artic.edu/artworks/{a['id']}",
            f"{base}/full/3840,/0/default.jpg",
            f"{base}/full/400,/0/default.jpg",
            thumb.get("width"),
            thumb.get("height"),
            tags=(a.get("subject_titles") or []) + (a.get("term_titles") or []),
        )

    @staticmethod
    def _fallbacks(it):
        # The image server may refuse very large sizes; try smaller ones before giving up.
        base = it["image_url"].rsplit("/full/", 1)[0]
        return [f"{base}/full/{w},/0/default.jpg" for w in (2400, 1686, 843)]

    async def _query(self, session, query, limit, page):
        params = {
            "q": query or "landscape",
            "query[term][is_public_domain]": "true",
            "fields": AIC_FIELDS,
            "limit": str(limit),
            "page": str(page + 1),
        }
        return await get_json(session, f"{self.api}/artworks/search?{urlencode(params)}")

    async def candidates(self, session, query):
        data = await self._query(session, query, 100, 0)
        return [
            i for i in (self._item(a) for a in data.get("data") or []) if i and i["landscape"] is not False
        ]

    async def resolve(self, session, it):
        return artwork_from(it, self._fallbacks(it))

    async def by_id(self, session, ident):
        data = await get_json(session, f"{self.api}/artworks/{ident}?{urlencode({'fields': AIC_FIELDS})}")
        it = self._item(data.get("data") or {})
        return artwork_from(it, self._fallbacks(it)) if it else None

    async def search(self, session, query, page, per_page):
        data = await self._query(session, query, per_page, page)
        items = [i for i in (self._item(a) for a in data.get("data") or []) if i]
        total = int((data.get("pagination") or {}).get("total") or len(items))
        return _page(items, page, per_page, total)


# -- National Gallery of Art -------------------------------------------------------

NGA_DATA = "https://github.com/NationalGalleryOfArt/opendata/raw/refs/heads/main/data"
NGA_TTL = 7 * 24 * 3600


class NGA:
    """No search API: the daily CSV export is downloaded weekly and indexed locally.

    Only paintings with an open-access primary image are kept (a few thousand rows, ~1 MB),
    so searching is a local word match over title, artist and medium (all words must match).
    """

    name = "nga"

    def __init__(self, cache_dir: Path, data_url: str = NGA_DATA) -> None:
        self.cache = cache_dir / "nga-index.json"
        self.data_url = data_url.rstrip("/")
        self._index: list[dict] | None = None
        self._lock = asyncio.Lock()

    async def index(self, session) -> list[dict]:
        async with self._lock:
            if self._index is None and self.cache.exists():
                try:
                    self._index = json.loads(self.cache.read_text())
                except (OSError, json.JSONDecodeError):
                    self._index = None
            fresh = self.cache.exists() and time.time() - self.cache.stat().st_mtime < NGA_TTL
            if self._index is not None and fresh:
                return self._index
            try:
                index = await self._build(session)
            except (aiohttp.ClientError, TimeoutError, ValueError, OSError) as e:
                if self._index is not None:
                    log.warning("NGA: refreshing the open-data index failed, using the old one (%s)", e)
                    return self._index
                raise
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(index))
            tmp.replace(self.cache)
            self._index = index
            log.info("NGA: indexed %d open-access paintings", len(index))
            return index

    async def _download(self, session, name) -> str:
        timeout = aiohttp.ClientTimeout(total=600)
        async with session.get(f"{self.data_url}/{name}", timeout=timeout) as resp:
            resp.raise_for_status()
            return (await resp.read()).decode("utf-8")

    async def _build(self, session) -> list[dict]:
        objects_csv = await self._download(session, "objects.csv")
        images_csv = await self._download(session, "published_images.csv")
        return await asyncio.to_thread(build_nga_index, objects_csv, images_csv)

    async def candidates(self, session, query):
        return [i for i in nga_match(await self.index(session), query) if i["landscape"] is not False]

    async def resolve(self, session, it):
        return artwork_from(it)

    async def by_id(self, session, ident):
        for it in await self.index(session):
            if it["id"] == str(ident):
                return artwork_from(it)
        return None

    async def search(self, session, query, page, per_page):
        matches = nga_match(await self.index(session), query)
        return _page(matches[page * per_page : (page + 1) * per_page], page, per_page, len(matches))


def build_nga_index(objects_csv: str, images_csv: str) -> list[dict]:
    paintings = {}
    for r in csv.DictReader(io.StringIO(objects_csv)):
        if r.get("classification") == "Painting":
            paintings[r["objectid"]] = r
    index = []
    for r in csv.DictReader(io.StringIO(images_csv)):
        o = paintings.get(r.get("depictstmsobjectid") or "")
        if not o or r.get("viewtype") != "primary" or r.get("openaccess") != "1" or not r.get("iiifurl"):
            continue
        width, height = int(r.get("width") or 0), int(r.get("height") or 0)
        size = min(width, 3840) if width else 3840
        it = item(
            "nga",
            o["objectid"],
            o.get("title"),
            o.get("attribution"),
            o.get("displaydate"),
            f"https://www.nga.gov/collection/art-object-page.{o['objectid']}.html",
            f"{r['iiifurl']}/full/{size},/0/default.jpg",
            r.get("iiifthumburl") or f"{r['iiifurl']}/full/!400,400/0/default.jpg",
            width,
            height,
        )
        it["medium"] = o.get("medium") or ""
        index.append(it)
    return index


def nga_match(index: list[dict], query: str) -> list[dict]:
    words = [w for w in re.split(r"\W+", (query or "").lower()) if w]
    if not words:
        return list(index)
    out = []
    for it in index:
        hay = f"{it['title']} {it['artist']} {it.get('medium', '')}".lower()
        if all(w in hay for w in words):
            out.append(it)
    return out


# -- Rijksmuseum ----------------------------------------------------------------------

AAT_ENGLISH = "http://vocab.getty.edu/aat/300388277"


class Rijks:
    """Linked Art search, then object → VisualItem → DigitalObject → iiif.micr.io image id."""

    name = "rijks"

    def __init__(self, api: str = "https://data.rijksmuseum.nl") -> None:
        self.api = api.rstrip("/")
        self._lists: dict[str, dict] = {}

    def _data_url(self, ident_url: str) -> str:
        path = ident_url.split("rijksmuseum.nl/", 1)[-1]
        return f"{self.api}/{path}?_profile=la-framed"

    async def _ids(self, session, query, token=None):
        params = {"type": "painting", "imageAvailable": "true"}
        if query:
            params["title"] = query
        if token:
            params["pageToken"] = token
        data = await get_json(session, f"{self.api}/search/collection?{urlencode(params)}")
        ids = [o["id"].rsplit("/", 1)[-1] for o in data.get("orderedItems") or [] if o.get("id")]
        nxt = (data.get("next") or {}).get("id") or ""
        m = re.search(r"pageToken=([^&]+)", nxt)
        total = int((data.get("partOf") or {}).get("totalItems") or len(ids))
        return ids, (m.group(1) if m else None), total

    async def candidates(self, session, query):
        ids, _, _ = await self._ids(session, query)
        return [{"key": f"rijks:{i}", "source": "rijks", "id": i} for i in ids]

    async def resolve(self, session, it):
        return await self.by_id(session, it["id"])

    async def _item(self, session, ident):
        obj = await get_json(session, self._data_url(f"rijksmuseum.nl/{ident}"))
        title = _rijks_name(obj)
        prod = obj.get("produced_by") or {}
        artist = ""
        for part in [prod, *(prod.get("part") or [])]:
            for who in part.get("carried_out_by") or []:
                artist = artist or who.get("_label") or ""
        for st in prod.get("referred_to_by") or []:
            artist = artist or st.get("content") or ""
        span = prod.get("timespan") or {}
        date = next((n.get("content") for n in span.get("identified_by") or [] if n.get("content")), "")
        end = str(span.get("end_of_the_end") or "")[:4]
        if end.isdigit() and int(end) > 1925:
            return None  # likely still in copyright
        visual = next((v["id"] for v in obj.get("shows") or [] if v.get("id")), None)
        if not visual:
            return None
        vis = await get_json(session, self._data_url(visual))
        digital = next((d["id"] for d in vis.get("digitally_shown_by") or [] if d.get("id")), None)
        if not digital:
            return None
        dig = await get_json(session, self._data_url(digital))
        access = next(
            (a["id"] for a in dig.get("access_point") or [] if "iiif.micr.io/" in a.get("id", "")), None
        )
        if not access:
            return None
        image_id = access.split("iiif.micr.io/", 1)[1].split("/", 1)[0]
        base = f"https://iiif.micr.io/{image_id}"
        return item(
            "rijks",
            ident,
            title,
            artist,
            date,
            f"https://id.rijksmuseum.nl/{ident}",
            f"{base}/full/3840,/0/default.jpg",
            f"{base}/full/400,/0/default.jpg",
        )

    async def by_id(self, session, ident):
        it = await self._item(session, ident)
        if not it:
            return None
        base = it["image_url"].rsplit("/full/", 1)[0]
        return artwork_from(it, [f"{base}/full/max/0/default.jpg"])

    async def search(self, session, query, page, per_page):
        # The API pages by opaque tokens; keep the ids seen so far per query and slice them.
        # Each item needs three more requests, so pages are kept small.
        per_page = min(per_page, 12)
        st = self._lists.setdefault(query, {"ids": [], "next": None, "done": False, "total": 0})
        while len(st["ids"]) < (page + 1) * per_page and not st["done"]:
            ids, nxt, total = await self._ids(session, query, st["next"])
            st["ids"] += ids
            st["next"], st["total"], st["done"] = nxt, total, not nxt or not ids
        chunk = st["ids"][page * per_page : (page + 1) * per_page]
        sem = asyncio.Semaphore(4)

        async def one(i):
            async with sem:
                try:
                    return await self._item(session, i)
                except (aiohttp.ClientError, TimeoutError, ValueError, KeyError):
                    return None

        items = [i for i in await asyncio.gather(*(one(i) for i in chunk)) if i]
        more = len(st["ids"]) > (page + 1) * per_page or not st["done"]
        return {"items": items, "total": st["total"], "page": page, "more": more}


def _rijks_name(obj: dict) -> str:
    names = [n for n in obj.get("identified_by") or [] if n.get("type") == "Name" and n.get("content")]
    for n in names:
        if any(lang.get("id") == AAT_ENGLISH for lang in n.get("language") or []):
            return n["content"]
    return names[0]["content"] if names else (obj.get("_label") or "Untitled")
