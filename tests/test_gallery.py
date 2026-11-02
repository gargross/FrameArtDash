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

    async def pick(self, query, sources, recent, min_width, size, exclude=()):
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


async def test_search_pages_and_items(tmp_path):
    g, server, seen = await _gallery(tmp_path)
    try:
        met = await g.search("met", "harbour", 0, per_page=2)
        assert [i["key"] for i in met["items"]] == ["met:1", "met:2"] and met["more"] is True
        assert met["items"][0]["thumb"].endswith("/img/wide.jpg")
        page2 = await g.search("met", "harbour", 1, per_page=2)
        assert [i["key"] for i in page2["items"]] == ["met:4"]  # met:3 is not public domain
        cma = await g.search("cleveland", "fields", 0)
        assert [i["landscape"] for i in cma["items"]] == [True, False]
        assert seen[-1]["skip"] == "0" and seen[-1]["limit"] == "24"
    finally:
        await server.close()


async def test_fetch_by_key_allows_hand_picked_portrait(tmp_path):
    g, server, _ = await _gallery(tmp_path)
    try:
        art, path = await g.fetch("met:4", (384, 216))
        assert art.title == "Portrait" and path.name == "met-4.jpg"
        try:
            await g.fetch("met:3", (384, 216))
            raise AssertionError("copyrighted work must be refused")
        except GalleryError:
            pass
    finally:
        await server.close()


def test_library_collection_and_schedule(tmp_path):
    from frameartdash.config import ConfigError
    from frameartdash.gallery import Library

    lib = Library(tmp_path)
    items = lib.save_collection(
        [{"key": "met:1", "title": "A"}, {"key": "met:1"}, {"key": "cleveland:9", "x": 1}]
    )
    assert [i["key"] for i in items] == ["met:1", "cleveland:9"] and "x" not in items[1]
    try:
        lib.save_collection([{"key": "louvre:1"}])
        raise AssertionError("unknown source must be refused")
    except ValueError:
        pass

    lib.save_schedule(
        {
            "rules": [
                {"days": ["sat", "sun"], "start": "08:00", "end": "12:00", "source": "collection"},
                {"start": "18:00", "end": "06:00", "source": "search", "query": "nocturne"},
            ]
        }
    )
    assert lib.active_rule(datetime(2026, 10, 31, 9))["source"] == "collection"  # Saturday
    assert lib.active_rule(datetime(2026, 11, 2, 9)) is None  # Monday morning
    assert lib.active_rule(datetime(2026, 11, 2, 23))["query"] == "nocturne"
    try:
        lib.save_schedule({"rules": [{"start": "25:00", "end": "26:00"}]})
        raise AssertionError("bad times must be refused")
    except ConfigError:
        pass


async def test_schedule_collection_rotation_show_and_hold(tmp_path):
    now = [datetime(2026, 10, 31, 9)]  # Saturday
    svc, rendered, _ = _cards_service(tmp_path, lambda: now[0])

    fetched = []

    async def fetch(key, size, min_width=1280):
        fetched.append(key)
        source, _, ident = key.partition(":")
        art = Artwork(source, ident, f"Fav {ident}")
        return art, svc.gallery._store(art, _jpeg(64, 36))

    svc.gallery.fetch = fetch
    svc.library.save_collection([{"key": "met:7"}, {"key": "met:8"}])
    svc.library.save_schedule(
        {"rules": [{"days": ["sat"], "start": "08:00", "end": "12:00", "source": "collection"}]}
    )

    await svc.run_cycle()
    now[0] += timedelta(minutes=61)
    await svc.run_cycle()
    assert sorted(fetched) == ["met:7", "met:8"]  # both favourites before any repeat
    assert svc.gallery.picks == 0  # no museum search during the collection rule

    current = await svc.gallery_show("met:42")
    assert current["title"] == "Fav 42" and fetched[-1] == "met:42"

    svc.set_hold(True)
    now[0] += timedelta(hours=3)
    await svc.run_cycle()
    assert svc.store.load().gallery["id"] == "42"  # held: no rotation

    svc.set_hold(False)
    await svc.gallery_next()
    assert svc.gallery.picks == 1  # 12:01 is outside the rule: museum search


# -- new museums, subject filter, alternatives ---------------------------------

NGA_OBJECTS = (
    "objectid,title,displaydate,attribution,medium,classification\n"
    "46,Country House in a Park,c. 1675,Jacob van Ruisdael,oil on canvas,Painting\n"
    "47,The Adoration of the Magi,1480,Botticelli,tempera on panel,Painting\n"
    "48,Two Jugs,1850,Unknown,stoneware,Decorative Art\n"
    "49,Tall Lady,1880,Someone,oil on canvas,Painting\n"
)


def _new_museums_app():
    images = {"aic-big.jpg": _jpeg(3840, 2200), "nga.jpg": _jpeg(3840, 2400), "rijks.jpg": _jpeg(4000, 2500)}
    calls = []

    async def aic_search(request):
        calls.append(("aic", dict(request.query)))
        return web.json_response(
            {
                "pagination": {"total": 3},
                "data": [
                    {
                        "id": 1,
                        "title": "Water Lilies",
                        "artist_title": "Claude Monet",
                        "date_display": "1906",
                        "image_id": "img1",
                        "is_public_domain": True,
                        "artwork_type_title": "Painting",
                        "thumbnail": {"width": 4000, "height": 2400},
                        "subject_titles": ["water", "flowers"],
                    },
                    {
                        "id": 2,
                        "title": "Evening Scene",
                        "artist_title": "X",
                        "image_id": "img2",
                        "is_public_domain": True,
                        "artwork_type_title": "Painting",
                        "thumbnail": {"width": 4000, "height": 2400},
                        "subject_titles": ["Saints", "Christianity"],
                    },
                    {
                        "id": 3,
                        "title": "A Print",
                        "image_id": "img3",
                        "is_public_domain": True,
                        "artwork_type_title": "Print",
                    },
                ],
            }
        )

    async def aic_iiif(request):
        size = request.match_info["size"]
        calls.append(("aic-img", size))
        if size == "3840,":
            return web.Response(status=403)  # the server refuses very large sizes
        return web.Response(body=images["aic-big.jpg"], content_type="image/jpeg")

    async def nga_csv(request):
        name = request.match_info["name"]
        calls.append(("nga", name))
        base = str(request.url.origin())
        if name == "objects.csv":
            return web.Response(text=NGA_OBJECTS)
        rows = [
            "uuid,iiifurl,iiifthumburl,viewtype,width,height,openaccess,depictstmsobjectid",
            f"u1,{base}/nga-iiif/u1,{base}/nga-iiif/u1/thumb,primary,7741,6036,1,46",
            f"u2,{base}/nga-iiif/u2,,primary,5000,3000,1,47",
            f"u3,{base}/nga-iiif/u3,,primary,5000,3000,1,48",
            f"u4,{base}/nga-iiif/u4,,primary,5000,3000,0,49",
        ]
        return web.Response(text="\n".join(rows) + "\n")

    async def nga_iiif(request):
        return web.Response(body=images["nga.jpg"], content_type="image/jpeg")

    async def rijks_search(request):
        calls.append(("rijks", dict(request.query)))
        return web.json_response(
            {
                "partOf": {"totalItems": 1},
                "orderedItems": [{"id": "https://id.rijksmuseum.nl/200100988", "type": "HumanMadeObject"}],
            }
        )

    async def rijks_record(request):
        ident = request.match_info["ident"]
        records = {
            "200100988": {
                "identified_by": [
                    {
                        "type": "Name",
                        "content": "De Molen",
                        "language": [{"id": "http://vocab.getty.edu/aat/300388256"}],
                    },
                    {
                        "type": "Name",
                        "content": "The Mill",
                        "language": [{"id": "http://vocab.getty.edu/aat/300388277"}],
                    },
                ],
                "produced_by": {
                    "part": [{"carried_out_by": [{"_label": "Jacob van Ruisdael"}]}],
                    "timespan": {"identified_by": [{"content": "c. 1670"}], "end_of_the_end": "1670-12-31"},
                },
                "shows": [{"id": "https://id.rijksmuseum.nl/202100988", "type": "VisualItem"}],
            },
            "202100988": {
                "digitally_shown_by": [{"id": "https://id.rijksmuseum.nl/500900", "type": "DigitalObject"}]
            },
            "500900": {"access_point": [{"id": "https://iiif.micr.io/ABCDE/full/max/0/default.jpg"}]},
        }
        return web.json_response(records[ident])

    app = web.Application()
    app.router.add_get("/aic/artworks/search", aic_search)
    app.router.add_get("/aic-iiif/{image}/full/{size}/0/default.jpg", aic_iiif)
    app.router.add_get("/nga/{name}", nga_csv)
    app.router.add_get("/nga-iiif/{rest:.*}", nga_iiif)
    app.router.add_get("/rijks/search/collection", rijks_search)
    app.router.add_get("/rijks/{ident}", rijks_record)
    return app, calls


async def _new_gallery(tmp_path):
    app, calls = _new_museums_app()
    server = TestServer(app)
    await server.start_server()
    base = str(server.make_url(""))
    g = Gallery(
        tmp_path / "gallery",
        aic_api=base + "/aic",
        aic_iiif=base + "/aic-iiif",
        nga_data=base + "/nga",
        rijks_api=base + "/rijks",
        rng=random.Random(3),
    )
    return g, server, calls


async def test_aic_public_domain_paintings_with_size_fallback_and_exclusion(tmp_path):
    from frameartdash.config import DEFAULT_EXCLUDE

    exclude = tuple(w.strip() for w in DEFAULT_EXCLUDE.split(","))
    g, server, calls = await _new_gallery(tmp_path)
    try:
        page = await g.search("aic", "water", 0, exclude=exclude)
        assert [i["key"] for i in page["items"]] == ["aic:1"] and page[
            "hidden"
        ] == 1  # 'Saints' tag hidden, print dropped
        art, path = await g.pick("water", ("aic",), [], 3000, (384, 216), exclude)
        assert art.key == "aic:1" and art.artist == "Claude Monet"
        assert ("aic-img", "3840,") in calls and ("aic-img", "2400,") in calls  # refused, then the next size
    finally:
        await server.close()


async def test_nga_index_open_access_paintings_and_word_match(tmp_path):
    g, server, calls = await _new_gallery(tmp_path)
    try:
        page = await g.search("nga", "oil, tempera, stoneware", 0)
        assert sorted(i["key"] for i in page["items"]) == ["nga:46", "nga:47"]  # painting + open access only
        assert (await g.search("nga", "ruisdael park", 0))["items"][0]["title"] == "Country House in a Park"
        assert (await g.search("nga", "ruisdael magi", 0))["items"] == []  # all words must match
        art, path = await g.pick("park, magi", ("nga",), [], 3000, (384, 216), ("Adoration",))
        assert art.key == "nga:46" and path.exists()
        assert sum(1 for c in calls if c == ("nga", "objects.csv")) == 1  # cached index
        assert (tmp_path / "gallery" / "nga-index.json").exists()
    finally:
        await server.close()


async def test_rijks_linked_art_chain(tmp_path):
    g, server, calls = await _new_gallery(tmp_path)
    try:
        page = await g.search("rijks", "molen", 0)
        it = page["items"][0]
        assert it["title"] == "The Mill" and it["artist"] == "Jacob van Ruisdael" and it["date"] == "c. 1670"
        assert it["thumb"] == "https://iiif.micr.io/ABCDE/full/400,/0/default.jpg"
        assert calls[0] == ("rijks", {"type": "painting", "imageAvailable": "true", "title": "molen"})
    finally:
        await server.close()


def test_split_terms_and_excluded():
    from frameartdash.gallery import excluded, split_terms

    assert split_terms(" seascape, winter harbor ,, ") == ["seascape", "winter harbor"]
    words = ("Saint", "Angel", "Last Supper")
    assert excluded("Saints in a Landscape", [], words) == "Saint"
    assert excluded("View of Michelangelo's Garden", [], words) is None  # word prefix, not substring
    assert excluded("The Last Supper", [], words) == "Last Supper"
    assert excluded("Harbor", ["Angels"], words) == "Angel"
    assert excluded("Harbor", [], ()) is None


async def test_search_merges_comma_alternatives(tmp_path):
    g, server, seen = await _gallery(tmp_path)
    try:
        res = await g.search("cleveland", "fields, harbour", 0)
        assert sorted(q["q"] for q in seen) == ["fields", "harbour"]
        assert [i["key"] for i in res["items"]] == ["cleveland:10", "cleveland:11"]  # de-duplicated
    finally:
        await server.close()
