"""One render → compose → push → cleanup cycle, plus the scheduling loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from frameartdash import compose, frame as frame_mod, render as render_mod
from frameartdash.cards.store import LayoutStore, normalize_card
from frameartdash.config import Settings, in_window
from frameartdash.gallery import Gallery, GalleryError, Library
from frameartdash.history import History
from frameartdash.state import State, StateStore

log = logging.getLogger(__name__)

Renderer = Callable[[Settings], Awaitable[bytes]]
CardRenderer = Callable[[Settings, dict], Awaitable[bytes]]


async def render_cards(settings: Settings, layout: dict) -> bytes:
    from frameartdash.cards import page

    html, tz = await page.build(settings, layout, inline=True)
    return await render_mod.render_html(settings, html, tz)


FrameFactory = Callable[[Settings], Any]  # returns an async context manager yielding a Frame


@dataclass
class CycleResult:
    status: str  # uploaded | unchanged | skipped | rendered | error
    reason: str = ""
    content_id: str | None = None
    distance: int | None = None
    deleted: list[str] = field(default_factory=list)
    finished: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))

    def as_dict(self) -> dict:
        return asdict(self)


async def retry(op: Callable[[], Awaitable[Any]], attempts: int, what: str, base_delay: float = 2.0) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            return await op()
        except Exception as e:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("%s failed (%s), retry %d/%d in %.0fs", what, e, attempt, attempts - 1, delay)
            await asyncio.sleep(delay)


class Service:
    def __init__(
        self,
        settings: Settings,
        renderer: Renderer = render_mod.render,
        frame_factory: FrameFactory = frame_mod.connect,
        clock: Callable[[], datetime] = datetime.now,
        card_renderer: CardRenderer = render_cards,
        gallery: Gallery | None = None,
    ) -> None:
        self.settings = settings
        self.renderer = renderer
        self.card_renderer = card_renderer
        self.frame_factory = frame_factory
        self.clock = clock
        self.store = StateStore(settings.state_file)
        self.history = History(settings.data_dir / "history")
        self.gallery = gallery or Gallery(settings.data_dir / "gallery")
        self.library = Library(settings.data_dir / "gallery")
        self.controls_file = settings.data_dir / "controls.json"
        self.lock = asyncio.Lock()
        self.last_result: CycleResult | None = None
        self.last_success: str | None = None
        self.last_jpeg: bytes | None = None
        self.next_run: datetime | None = None
        self._wake = asyncio.Event()
        self._last_had_cards = True
        # Render a preview even if the TV is off: on first start and after settings changes.
        self.preview_pending = self.history.preview_info() is None

    def apply_settings(self, settings: Settings) -> None:
        """Use new settings from the next cycle on and reschedule right away."""
        self.settings = settings
        # The look may have changed even if the dashboard did not: skip the diff once.
        state = self.store.load()
        state.last_hash = None
        self.store.save(state)
        self.preview_pending = True
        self._wake.set()

    # -- cycle ---------------------------------------------------------------

    async def run_cycle(self, force: bool = False, upload: bool = True) -> CycleResult:
        """Run one cycle. ``force`` ignores the active hours and the diff check."""
        async with self.lock:
            try:
                result = await self._cycle(force=force, upload=upload)
            except Exception as e:
                log.error("Cycle failed: %s", e, exc_info=log.isEnabledFor(logging.DEBUG))
                result = CycleResult("error", reason=str(e) or type(e).__name__)
            self.last_result = result
            if result.status in ("uploaded", "unchanged", "rendered"):
                self.last_success = result.finished
            if result.status != "error":
                log.info("Cycle %s%s", result.status, f": {result.reason}" if result.reason else "")
            return result

    async def _cycle(self, force: bool, upload: bool) -> CycleResult:
        s = self.settings
        if not s.configured:
            return CycleResult("skipped", reason=s.not_configured_reason)

        state = self.store.load()
        if s.layout == "cards" and s.gallery:
            await self._rotate_gallery(state)
        # Cards that just expired must leave the screen even outside the active hours.
        if (
            s.layout == "cards"
            and s.cards_on_demand
            and state.cards_on_screen
            and not self.cards_wanted(state)
        ):
            force = True
        if not upload:
            await self._render_and_compose(state, force=True)
            return CycleResult("rendered", reason="preview, not uploaded")
        if s.dry_run:
            if not force and not in_window(s.active_hours, self.clock().time()):
                return CycleResult("skipped", reason="outside active hours")
            await self._render_and_compose(state, force=True)
            return CycleResult("rendered", reason="test mode, TV not contacted")

        if not s.frame_host:
            result = CycleResult("skipped", reason="no TV IP address configured")
        elif not force and not in_window(s.active_hours, self.clock().time()):
            result = CycleResult("skipped", reason="outside active hours")
        else:
            try:
                result = await self._push(state, force)
            except frame_mod.FrameUnreachable as e:
                result = CycleResult("skipped", reason=str(e))

        if result.status == "skipped" and self.preview_pending:
            await self._render_and_compose(state, force=True)
            result.reason += "; preview rendered"
        return result

    async def _push(self, state: State, force: bool) -> CycleResult:
        s = self.settings
        async with self.frame_factory(s) as frame:
            power = await frame.power_state()
            if power not in frame_mod.POWER_ON_STATES:
                return CycleResult("skipped", reason=f"TV is {power}")
            if s.only_in_artmode and power != "art":
                return CycleResult("skipped", reason=f"TV is not in art mode ({power})")

            image, jpeg, new_hash, distance = await self._render_and_compose(state, force=force)
            if jpeg is None:
                return CycleResult("unchanged", reason=f"hash distance {distance}", distance=distance)

            content_id = await retry(lambda: frame.upload(jpeg), s.retries, "Upload")
            log.info("Uploaded %s (%d KiB)", content_id, len(jpeg) // 1024)
            # Track it right away so a failing select does not leave an orphan behind.
            state.add(content_id)
            self.store.save(state)

            # Only show it right away while the TV is already in Art Mode, so live TV is never interrupted.
            await retry(lambda: frame.select(content_id, show=power == "art"), s.retries, "Select")
            state.current_id = content_id
            state.last_hash = new_hash
            state.cards_on_screen = self._last_had_cards
            now = self.clock().astimezone()
            state.last_upload = now.isoformat(timespec="seconds")
            self.store.save(state)
            await asyncio.to_thread(self.history.add, image, content_id, now)
            self.history.mark_preview_uploaded(content_id)

            deleted = await self._delete_stale(frame, state)
            return CycleResult("uploaded", content_id=content_id, distance=distance, deleted=deleted)

    async def _render_and_compose(self, state: State, force: bool):
        """Returns (image, jpeg, hash, distance); image and jpeg are None if unchanged."""
        s = self.settings
        started = time.monotonic()
        if s.layout == "cards":
            layout = LayoutStore(s.cards_file).load()
            self._last_had_cards = self.cards_wanted(state)
            if not self._last_had_cards:
                layout = {**layout, "cards": self._art_only_cards(state)}
            art = self.gallery.current_path(state.gallery) if s.gallery else None
            render_settings = replace(s, style="image", bg_image=str(art)) if art else s
            png = await retry(lambda: self.card_renderer(render_settings, layout), s.retries, "Render")
        else:
            png = await retry(lambda: self.renderer(s), s.retries, "Render")
        shot = compose.load(png)
        new_hash = compose.image_hash(shot)
        distance = compose.hash_distance(state.last_hash, new_hash)
        if not force and s.diff_threshold > 0 and distance is not None and distance < s.diff_threshold:
            return None, None, new_hash, distance

        if s.layout == "cards":
            # The card page already is the finished picture.
            image = shot if shot.size == s.output_size else shot.resize(s.output_size)
        else:
            image = await asyncio.to_thread(compose.compose, shot, s)
        jpeg = await asyncio.to_thread(compose.to_jpeg, image, s.jpeg_quality)
        self.last_jpeg = jpeg
        self.preview_pending = False
        info = {
            "time": self.clock().astimezone().isoformat(timespec="seconds"),
            "seconds": round(time.monotonic() - started, 1),
            "url": s.target_url,
            "size_kb": len(jpeg) // 1024,
        }
        await asyncio.to_thread(self.history.save_preview, jpeg, png, info)
        if s.debug_images:
            self._save_debug(png, jpeg)
        return image, jpeg, new_hash, distance

    # -- cards on demand ----------------------------------------------------

    # Cards-until and hold live in their own file: they are changed from the web API while a
    # cycle may be running, and a cycle saving state.json must not overwrite them.
    def controls(self) -> dict:
        try:
            data = json.loads(self.controls_file.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        return {"cards_until": data.get("cards_until"), "hold": bool(data.get("hold", False))}

    def _save_controls(self, **changes) -> dict:
        data = {**self.controls(), **changes}
        self.controls_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.controls_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.controls_file)
        return data

    def cards_wanted(self, state: State | None = None) -> bool:
        """Whether the picture should carry the cards right now."""
        if not self.settings.cards_on_demand:
            return True
        until = self.controls()["cards_until"]
        if not until:
            return False
        try:
            return self.clock().astimezone() < datetime.fromisoformat(until)
        except ValueError:
            return False

    def show_cards(self, minutes: int | None = None) -> str:
        """Show the cards for ``minutes`` (default CARDS_MINUTES) from now. Returns the end time."""
        minutes = max(1, int(minutes or self.settings.cards_minutes))
        until = (self.clock().astimezone() + timedelta(minutes=minutes)).isoformat(timespec="seconds")
        self._save_controls(cards_until=until)
        self._wake.set()
        return until

    def hide_cards(self) -> None:
        self._save_controls(cards_until=None)
        self._wake.set()

    def set_hold(self, hold: bool) -> None:
        """Keep the current artwork (no rotation) until released."""
        self._save_controls(hold=bool(hold))
        self._wake.set()

    def _art_only_cards(self, state: State) -> list[dict]:
        art = state.gallery if self.settings.gallery else None
        if not (art and self.settings.gallery_caption):
            return []
        caption = art.get("title", "")
        who = ", ".join(p for p in (art.get("artist"), art.get("date")) if p)
        if who:
            caption = f"{caption} — {who}" if caption else who
        card = {
            "id": "caption",
            "type": "text",
            "x": 2,
            "y": 93,
            "w": 70,
            "h": 5,
            "options": {"text": caption, "style": "none", "scale": 55},
        }
        return [normalize_card(card)]

    # -- gallery -------------------------------------------------------------

    def gallery_due(self, state: State) -> bool:
        if self.gallery.current_path(state.gallery) is None:
            return True
        try:
            chosen = datetime.fromisoformat(state.gallery["chosen_at"])
        except (KeyError, TypeError, ValueError):
            return True
        return self.clock().astimezone() - chosen >= timedelta(minutes=self.settings.gallery_interval_min)

    async def _rotate_gallery(self, state: State, force: bool = False) -> None:
        """Pick a new artwork when one is due (or ``force``). Failures keep the current one."""
        if not force and (self.controls()["hold"] or not self.gallery_due(state)):
            if self.gallery.current_path(state.gallery) is not None:
                return
        s = self.settings
        rule = self.library.active_rule(self.clock().astimezone())
        try:
            if rule and rule["source"] == "collection" and self.library.collection():
                items = self.library.collection()
                n = len(items) - 1  # avoid the last n picks, so every item comes round before a repeat
                recent = set(state.gallery_recent[-n:]) if n > 0 else set()
                fresh = [i for i in items if i["key"] not in recent]
                key = random.choice(fresh or items)["key"]
                art, path = await self.gallery.fetch(key, s.output_size)
            else:
                query = (rule or {}).get("query") or s.gallery_query
                art, path = await self.gallery.pick(
                    query, s.gallery_sources, state.gallery_recent, s.gallery_min_width, s.output_size
                )
        except (GalleryError, OSError) as e:
            log.warning("Gallery: keeping the current picture, no new artwork (%s)", e)
            return
        self._remember_artwork(state, art, path)

    def _remember_artwork(self, state: State, art, path) -> None:
        state.gallery = {
            **art.as_dict(),
            "file": path.name,
            "chosen_at": self.clock().astimezone().isoformat(timespec="seconds"),
        }
        state.gallery_recent = [k for k in state.gallery_recent if k != art.key][-199:] + [art.key]
        state.last_hash = None  # a new painting always goes to the TV
        self.store.save(state)

    async def gallery_show(self, key: str) -> dict:
        """Make one hand-picked artwork the current painting (it stays for a full interval)."""
        async with self.lock:
            state = self.store.load()
            art, path = await self.gallery.fetch(key, self.settings.output_size)
            self._remember_artwork(state, art, path)
            return state.gallery

    async def gallery_next(self) -> dict | None:
        """Skip to the next artwork now, following the schedule."""
        async with self.lock:
            state = self.store.load()
            await self._rotate_gallery(state, force=True)
            return state.gallery

    def next_wake_seconds(self) -> float:
        """Seconds until the next cycle: the interval, or sooner when cards or the artwork expire."""
        s = self.settings
        wait = float(s.interval_min * 60)
        state = self.store.load()
        now = self.clock().astimezone()
        controls = self.controls()
        if s.layout == "cards" and s.cards_on_demand and controls["cards_until"] and state.cards_on_screen:
            try:
                left = (datetime.fromisoformat(controls["cards_until"]) - now).total_seconds() + 1
                wait = min(wait, max(1.0, left))
            except ValueError:
                pass
        if (
            s.layout == "cards"
            and s.gallery
            and not controls["hold"]
            and state.gallery
            and state.gallery.get("chosen_at")
        ):
            try:
                due = datetime.fromisoformat(state.gallery["chosen_at"]) + timedelta(
                    minutes=s.gallery_interval_min
                )
                wait = min(wait, max(1.0, (due - now).total_seconds() + 1))
            except ValueError:
                pass
        return wait

    def _save_debug(self, png: bytes, jpeg: bytes) -> None:
        try:
            self.settings.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.debug_dir / "screenshot.png").write_bytes(png)
            (self.settings.debug_dir / "frame.jpg").write_bytes(jpeg)
        except OSError as e:
            log.warning("Could not write debug images: %s", e)

    # -- cleanup -------------------------------------------------------------

    async def _delete_stale(self, frame, state: State) -> list[str]:
        """Delete every managed image except the selected one. Never raises."""
        stale = state.stale_ids()
        if self.settings.cleanup_unmanaged:
            try:
                for cid in await frame.my_photos():
                    if cid != state.current_id and cid not in stale:
                        stale.append(cid)
            except Exception as e:
                log.warning("Listing My Photos failed: %s", e)
        if not stale:
            return []
        try:
            await frame.delete(stale)
        except Exception as e:
            log.warning("Deleting %s failed, will retry next cycle: %s", ", ".join(stale), e)
            return []
        state.forget(stale)
        self.store.save(state)
        log.info("Deleted %s", ", ".join(stale))
        return stale

    async def cleanup(self) -> list[str]:
        """Remove orphaned images left over from earlier runs."""
        async with self.lock:
            state = self.store.load()
            if not self.settings.frame_host:
                return []
            if not state.stale_ids() and not self.settings.cleanup_unmanaged:
                return []
            async with self.frame_factory(self.settings) as frame:
                if (await frame.power_state()) not in frame_mod.POWER_ON_STATES:
                    log.info("TV is not on, skipping startup cleanup")
                    return []
                return await self._delete_stale(frame, state)

    # -- loop ----------------------------------------------------------------

    async def run_forever(self) -> None:
        try:
            await self.cleanup()
        except frame_mod.FrameUnreachable as e:
            log.info("Startup cleanup skipped: %s", e)
        except Exception as e:
            log.warning("Startup cleanup failed: %s", e)
        while True:
            self._wake.clear()
            await self.run_cycle()
            interval = self.next_wake_seconds()
            self.next_run = self.clock() + timedelta(seconds=interval)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=interval)
            except TimeoutError:
                pass
