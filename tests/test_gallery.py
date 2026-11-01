import io
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image
from test_service import FakeFrame, _png

from frameartdash.config import Settings
from frameartdash.gallery import Artwork, Gallery, GalleryError
from frameartdash.service import Service


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 60, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _museum_app():
    """Fake Met and Cleveland APIs plus their image hosts."""
    images = {
        "wide.jpg": _jpeg(4000, 2400),
        "small.jpg": _jpeg(1200, 700),
        "tall.jpg": _jpeg(3000, 4000),
        "cma-wide.jpg": _jpeg(3600, 2200),
    }
    met_objects = {
        "1": {
            "isPublicDomain": True,
            "primaryImage": "/img/wide.jpg",
            "title": "Harbour",
            "artistDisplayName": "A",
        },
        "2": {"isPublicDomain": True, "primaryImage": "/img/small.jpg", "title": "Small"},
        "3": {"isPublicDomain": False, "primaryImage": "/img/wide.jpg", "title": "Copyrighted"},
        "4": {"isPublicDomain": True, "primaryImage": "/img/tall.jpg", "title": "Portrait"},
    }
    seen = []

    async def met_search(request):
        seen.append(dict(request.query))
        return web.json_response({"total": 4, "objectIDs": [1, 2, 3, 4]})

    async def met_object(request):
        o = met_objects[request.match_info["id"]]
        base = str(request.url.origin())
        return web.json_response({**o, "primaryImage": base + o["primaryImage"], "objectDate": "1872"})

    async def cma(request):
        seen.append(dict(request.query))
        base = str(request.url.origin())
        return web.json_response(
            {
                "data": [
                    {
                        "id": 10,
                        "title": "Fields",
                        "creation_date": "1890",
                        "share_license_status": "CC0",
                        "creators": [{"description": "B (French, 1840–1900)"}],
                        "images": {
                            "print": {"url": base + "/img/cma-wide.jpg", "width": "3600", "height": "2200"}
                        },
                    },
                    {
                        "id": 11,
                        "title": "Tall",
                        "share_license_status": "CC0",
                        "images": {
                            "print": {"url": base + "/img/tall.jpg", "width": "3000", "height": "4000"}
                        },
                    },
                ]
            }
        )

    async def image(request):
        return web.Response(body=images[request.match_info["name"]], content_type="image/jpeg")

    app = web.Application()
    app.router.add_get("/met/search", met_search)
    app.router.add_get("/met/objects/{id}", met_object)
    app.router.add_get("/cma/", cma)
    app.router.add_get("/img/{name}", image)
    return app, seen


async def _gallery(tmp_path, seed=1):
    app, seen = _museum_app()
    server = TestServer(app)
    await server.start_server()
    base = str(server.make_url(""))
    g = Gallery(tmp_path / "gallery", met_api=base + "/met", cma_api=base + "/cma/", rng=random.Random(seed))
    return g, server, seen


async def test_met_picks_only_public_domain_landscape_large_enough(tmp_path):
    g, server, seen = await _gallery(tmp_path)
    try:
        for _ in range(3):
            art, path = await g.pick("nocturne", ("met",), [], 3000, (384, 216))
            assert art.key == "met:1" and art.title == "Harbour" and art.date == "1872"
            with Image.open(path) as img:
                assert img.size == (384, 216)
        assert seen[0]["q"] == "nocturne" and seen[0]["medium"] == "Paintings"
        assert [p.name for p in (tmp_path / "gallery").glob("*.jpg")] == ["met-1.jpg"]
    finally:
        await server.close()


async def test_recent_picks_are_not_repeated(tmp_path):
    g, server, _ = await _gallery(tmp_path)
    try:
        try:
            await g.pick("x", ("met",), ["met:1"], 3000, (384, 216))
            raise AssertionError("expected GalleryError: the only suitable Met work was recent")
        except GalleryError:
            pass
    finally:
        await server.close()


async def test_cleveland_skips_portrait_and_parses_artist(tmp_path):
    g, server, seen = await _gallery(tmp_path)
    try:
        art, path = await g.pick("fields", ("cleveland",), [], 3000, (384, 216))
        assert art.key == "cleveland:10" and art.artist == "B" and art.caption == "Fields — B, 1890"
        assert seen[0]["cc0"] == "1" and seen[0]["type"] == "Painting"
        assert path.name == "cleveland-10.jpg"
    finally:
        await server.close()


class FakeGallery(Gallery):
    def __init__(self, directory):
        super().__init__(directory)
        self.picks = 0

    async def pick(self, query, sources, recent, min_width, size):
        self.picks += 1
        art = Artwork("met", str(self.picks), f"Painting {self.picks}", "Artist", "1870")
        return art, self._store(art, _jpeg(64, 36))


def _cards_service(tmp_path, clock, **env):
    settings = Settings.from_env(
        {
            "LAYOUT": "cards",
            "HA_URL": "http://ha",
            "HA_TOKEN": "t",
            "FRAME_HOST": "tv",
            "DATA_DIR": str(tmp_path),
            "RETRIES": "1",
            "GALLERY": "true",
            "CARDS_ON_DEMAND": "true",
            "CARDS_MINUTES": "15",
            **env,
        }
    )
    rendered = []

    async def card_renderer(s, layout):
        rendered.append((s, layout))
        return _png(f"{len(layout['cards'])} {s.bg_image.rsplit('/', 1)[-1]}")

    fake = FakeFrame()

    @asynccontextmanager
    async def factory(_settings):
        yield fake

    svc = Service(
        settings,
        frame_factory=factory,
        card_renderer=card_renderer,
        clock=clock,
        gallery=FakeGallery(tmp_path / "gallery"),
    )
    return svc, rendered, fake


async def test_art_alone_until_cards_requested_then_expire(tmp_path):
    now = [datetime(2026, 1, 1, 12)]
    svc, rendered, fake = _cards_service(tmp_path, lambda: now[0])
    (tmp_path / "cards.json").write_text('{"version": 1, "cards": [{"type": "clock"}]}')

    first = await svc.run_cycle()
    s, layout = rendered[-1]
    assert first.status == "uploaded"
    assert [c["id"] for c in layout["cards"]] == ["caption"]  # art alone, with its caption
    assert "Painting 1 — Artist, 1870" in layout["cards"][0]["options"]["text"]
    assert s.style == "image" and s.bg_image.endswith("met-1.jpg")
    assert svc.store.load().cards_on_screen is False

    svc.show_cards()
    await svc.run_cycle(force=True)
    assert [c["type"] for c in rendered[-1][1]["cards"]] == ["clock"]
    assert svc.store.load().cards_on_screen is True
    assert 0 < svc.next_wake_seconds() <= 15 * 60 + 1

    # Expired at 23:30, outside the active hours: still replaced by the art.
    now[0] = datetime(2026, 1, 1, 23, 30)
    result = await svc.run_cycle()
    assert result.status == "uploaded"
    assert [c["id"] for c in rendered[-1][1]["cards"]] == ["caption"]
    assert svc.store.load().cards_on_screen is False


async def test_hide_cards_and_hourly_rotation(tmp_path):
    now = [datetime(2026, 1, 1, 12)]
    svc, rendered, _ = _cards_service(tmp_path, lambda: now[0], GALLERY_CAPTION="false")
    await svc.run_cycle()
    assert rendered[-1][1]["cards"] == []
    svc.show_cards(5)
    svc.hide_cards()
    assert svc.cards_wanted(svc.store.load()) is False

    now[0] += timedelta(minutes=30)
    await svc.run_cycle()
    assert svc.gallery.picks == 1
    assert 0 < svc.next_wake_seconds() <= 30 * 60 + 1

    now[0] += timedelta(minutes=31)
    await svc.run_cycle()
    assert svc.gallery.picks == 2
    assert rendered[-1][0].bg_image.endswith("met-2.jpg")
    assert svc.store.load().gallery_recent == ["met:1", "met:2"]


async def test_cards_always_without_on_demand(tmp_path):
    svc, rendered, _ = _cards_service(tmp_path, lambda: datetime(2026, 1, 1, 12), CARDS_ON_DEMAND="false")
    (tmp_path / "cards.json").write_text('{"version": 1, "cards": [{"type": "clock"}]}')
    await svc.run_cycle()
    assert [c["type"] for c in rendered[-1][1]["cards"]] == ["clock"]
