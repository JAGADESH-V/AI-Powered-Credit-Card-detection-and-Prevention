"""Redis-backed state.

Every store here has an in-memory fallback so the project runs with no
infrastructure, and a Redis implementation that is correct across multiple
instances. The fallback is chosen explicitly at startup and logged — a
silent downgrade to per-process state would mean rate limits and replay
protection quietly stop working the moment you scale past one pod.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict, deque
from datetime import UTC, datetime
from typing import Protocol

from fraudguard.features.engineering import CardHistory
from fraudguard.observability import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard
    import redis as redis_lib
except ImportError:  # pragma: no cover
    redis_lib = None


def connect(url: str):
    """Return a live Redis client, or None if unavailable."""
    if redis_lib is None:
        logger.warning("redis.unavailable", extra={"reason": "package not installed"})
        return None
    try:
        client = redis_lib.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        client.ping()
        logger.info("redis.connected")
        return client
    except Exception as exc:
        logger.warning("redis.unavailable", extra={"reason": type(exc).__name__})
        return None


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------
class RateLimiter(Protocol):
    def allow(self, identity: str, limit: int, window_seconds: int) -> tuple[bool, int]: ...


class InMemoryRateLimiter:
    """Correct for a single process only."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, identity: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        import time

        now = time.monotonic()
        window = self._hits[identity]
        while window and now - window[0] > window_seconds:
            window.popleft()
        if len(window) >= limit:
            return False, 0
        window.append(now)
        return True, limit - len(window)


class RedisRateLimiter:
    """Sliding window over a sorted set, applied atomically.

    The whole trim-count-add sequence runs server-side in Lua. Doing it as
    separate round trips would let concurrent requests each observe a
    below-limit count and all be admitted.
    """

    SCRIPT = """
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local member = ARGV[4]
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
    local count = redis.call('ZCARD', key)
    if count >= limit then
      return {0, 0}
    end
    redis.call('ZADD', key, now, member)
    redis.call('EXPIRE', key, math.ceil(window))
    return {1, limit - count - 1}
    """

    def __init__(self, client) -> None:
        self._client = client
        self._script = client.register_script(self.SCRIPT)

    def allow(self, identity: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        import time
        import uuid

        now = time.time()
        try:
            allowed, remaining = self._script(
                keys=[f"rl:{identity}"],
                args=[now, window_seconds, limit, f"{now}:{uuid.uuid4().hex[:8]}"],
            )
            return bool(allowed), int(remaining)
        except Exception:
            # Fail open on infrastructure failure: rate limiting is a
            # availability control, and taking payments offline because Redis
            # blipped is worse than briefly unmetered traffic. Replay
            # protection below deliberately makes the opposite choice.
            logger.error("ratelimit.backend_error")
            return True, limit


# ---------------------------------------------------------------------------
# replay / nonce
# ---------------------------------------------------------------------------
class InMemoryNonceStore:
    def __init__(self, max_size: int = 100_000) -> None:
        self._seen: deque[str] = deque(maxlen=max_size)
        self._set: set[str] = set()

    def check_and_store(self, nonce: str, ttl_seconds: int) -> bool:
        if nonce in self._set:
            return False
        if len(self._seen) == self._seen.maxlen and self._seen:
            self._set.discard(self._seen[0])
        self._seen.append(nonce)
        self._set.add(nonce)
        return True


class RedisNonceStore:
    """SET NX is atomic, so two concurrent replays cannot both win."""

    def __init__(self, client) -> None:
        self._client = client

    def check_and_store(self, nonce: str, ttl_seconds: int) -> bool:
        try:
            return bool(self._client.set(f"nonce:{nonce}", "1", nx=True, ex=ttl_seconds))
        except Exception:
            # Fail closed: if replay protection cannot be verified, reject.
            # An unverifiable request is not one to trust with a payment.
            logger.error("nonce.backend_error")
            return False


# ---------------------------------------------------------------------------
# card history
# ---------------------------------------------------------------------------
class InMemoryCardHistoryStore:
    def __init__(self, max_cards: int = 200_000, retention_hours: int = 48) -> None:
        self._cache: OrderedDict[str, CardHistory] = OrderedDict()
        self.max_cards = max_cards
        self.retention_hours = retention_hours

    def get(self, card_token: str) -> CardHistory:
        history = self._cache.get(card_token)
        if history is None:
            history = CardHistory.empty()
            self._cache[card_token] = history
        self._cache.move_to_end(card_token)
        while len(self._cache) > self.max_cards:
            self._cache.popitem(last=False)
        return history

    def record(self, card_token: str, txn: dict, *, trusted: bool = True) -> None:
        history = self.get(card_token)
        history.record(
            txn["occurred_at"],
            float(txn["amount"]),
            (txn["latitude"], txn["longitude"]) if txn.get("latitude") is not None else None,
            txn["merchant_id"],
            txn.get("device_id"),
            trusted=trusted,
        )
        self._prune(history, txn["occurred_at"])

    def _prune(self, history: CardHistory, now: datetime) -> None:
        from datetime import timedelta

        cutoff = now - timedelta(hours=self.retention_hours)
        keep = [i for i, ts in enumerate(history.timestamps) if ts >= cutoff]
        if len(keep) == len(history.timestamps):
            return
        history.timestamps = [history.timestamps[i] for i in keep]
        history.amounts = [history.amounts[i] for i in keep]
        history.coordinates = [history.coordinates[i] for i in keep]
        history.merchants = [history.merchants[i] for i in keep]
        history.trusted = [history.trusted[i] for i in keep]


class RedisCardHistoryStore:
    """Per-card rolling window in a sorted set scored by timestamp.

    Keyed by card token, never by PAN. TTL matches the longest feature window,
    so memory is bounded and stale history expires without a sweeper job.
    """

    def __init__(self, client, retention_hours: int = 48) -> None:
        self._client = client
        self.retention_hours = retention_hours
        self._ttl = int(retention_hours * 3600)

    def _key(self, card_token: str) -> str:
        return f"hist:{card_token}"

    def get(self, card_token: str) -> CardHistory:
        history = CardHistory.empty()
        try:
            cutoff = datetime.now(UTC).timestamp() - self._ttl
            raw = self._client.zrangebyscore(self._key(card_token), cutoff, "+inf")
        except Exception:
            logger.error("history.backend_error", extra={"op": "get"})
            return history

        for blob in raw:
            try:
                entry = json.loads(blob)
            except json.JSONDecodeError:
                continue
            history.record(
                datetime.fromtimestamp(entry["t"], tz=UTC),
                entry["a"],
                tuple(entry["c"]) if entry.get("c") else None,
                entry["m"],
                entry.get("d"),
                trusted=entry.get("k", True),
            )
        return history

    def record(self, card_token: str, txn: dict, *, trusted: bool = True) -> None:
        when: datetime = txn["occurred_at"]
        entry = {
            "t": when.timestamp(),
            "a": float(txn["amount"]),
            "c": [txn["latitude"], txn["longitude"]] if txn.get("latitude") is not None else None,
            "m": txn["merchant_id"],
            "d": txn.get("device_id"),
            "k": trusted,
        }
        key = self._key(card_token)
        try:
            pipe = self._client.pipeline()
            pipe.zadd(key, {json.dumps(entry, separators=(",", ":")): when.timestamp()})
            pipe.zremrangebyscore(key, 0, when.timestamp() - self._ttl)
            pipe.expire(key, self._ttl)
            pipe.execute()
        except Exception:
            logger.error("history.backend_error", extra={"op": "record"})


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------
class InMemoryIdempotencyStore:
    def __init__(self, max_entries: int = 100_000) -> None:
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self.max_entries = max_entries

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict, ttl_seconds: int = 86400) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class RedisIdempotencyStore:
    def __init__(self, client) -> None:
        self._client = client

    def get(self, key: str) -> dict | None:
        try:
            blob = self._client.get(f"idem:{key}")
            return json.loads(blob) if blob else None
        except Exception:
            return None

    def put(self, key: str, value: dict, ttl_seconds: int = 86400) -> None:
        try:
            self._client.set(
                f"idem:{key}", json.dumps(value, default=str), ex=ttl_seconds
            )
        except Exception:
            logger.error("idempotency.backend_error")
