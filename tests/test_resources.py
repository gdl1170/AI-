"""Tests per hycoder/resources.py."""

import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hycoder.resources import (
    DiskCache,
    ResourceMonitor,
    ManagedPool,
    ResourceManager,
    get_resource_manager,
    reset_resource_manager,
)


class TestResourceMonitor:
    def test_cpu_percent_returns_float(self):
        rm = ResourceMonitor()
        val = rm.cpu_percent()
        assert isinstance(val, float)
        assert val >= 0

    def test_memory_bytes_returns_int(self):
        rm = ResourceMonitor()
        val = rm.memory_bytes()
        assert isinstance(val, (int, float))
        assert val >= 0

    def test_memory_mb_returns_float(self):
        rm = ResourceMonitor()
        val = rm.memory_mb()
        assert isinstance(val, float)
        assert val >= 0

    def test_snapshot_has_all_keys(self):
        rm = ResourceMonitor()
        snap = rm.snapshot()
        assert "timestamp" in snap
        assert "uptime_s" in snap
        assert "process_memory_mb" in snap
        assert "cpu_percent" in snap
        assert "ollama" in snap
        assert "cpu_count" in snap

    def test_ollama_status_not_running(self):
        status = ResourceMonitor.ollama_status(base_url="http://localhost:1")
        assert status["running"] is False

    def test_snapshot_limits_samples(self):
        rm = ResourceMonitor()
        for _ in range(4000):
            rm.snapshot()
        assert len(rm._samples) <= 2200  # trim to 1800 when >3600, then ~399 more added


class TestManagedPool:
    def test_submit_and_active(self):
        pool = ManagedPool(max_workers=2)

        def dummy():
            time.sleep(0.1)

        t = pool.submit(dummy)
        assert isinstance(t, threading.Thread)
        time.sleep(0.05)
        assert pool.active_count >= 0
        t.join(timeout=1)

    def test_max_workers_default(self):
        pool = ManagedPool()
        assert pool.max_workers >= 1

    def test_wait_timeout(self):
        pool = ManagedPool(max_workers=2)

        def slow():
            time.sleep(0.2)

        pool.submit(slow)
        pool.wait(timeout=0.5)


class TestResourceManager:
    def test_init_sets_defaults(self):
        mgr = ResourceManager()
        assert mgr.max_memory_mb > 0
        assert mgr.max_cache_entries > 0
        assert mgr.max_conversation_messages > 0
        assert mgr.max_conversation_tokens > 0
        assert mgr.knowledge_max_chunks > 0

    def test_init_with_cfg(self):
        cfg = {
            "max_memory_mb": "256",
            "max_cache_entries": "512",
            "max_conversation_messages": "50",
        }
        mgr = ResourceManager(cfg)
        assert mgr.max_memory_mb == 256
        assert mgr.max_cache_entries == 512
        assert mgr.max_conversation_messages == 50

    def test_caches_created(self):
        mgr = ResourceManager()
        assert mgr.response_cache is not None
        assert mgr.embedding_cache is not None
        assert mgr.response_cache.max_entries == mgr.max_cache_entries

    def test_monitor_created(self):
        mgr = ResourceManager()
        assert mgr.monitor is not None

    def test_pool_created(self):
        mgr = ResourceManager()
        assert mgr.pool is not None

    def test_check_memory_returns_bool(self):
        mgr = ResourceManager()
        result = mgr.check_memory()
        assert isinstance(result, bool)

    def test_snapshot_has_limits(self):
        mgr = ResourceManager()
        snap = mgr.snapshot()
        assert "limits" in snap
        assert "max_memory_mb" in snap["limits"]
        assert "cache_entries" in snap["limits"]
        assert "pool_active" in snap["limits"]
        assert "gc_objects" in snap

    def test_cleanup_runs(self):
        mgr = ResourceManager()
        mgr.cleanup()

    def test_close_runs(self):
        mgr = ResourceManager()
        mgr.close()

    def test_singleton(self):
        mgr1 = get_resource_manager()
        mgr2 = get_resource_manager()
        assert mgr1 is mgr2

    def test_reset(self):
        mgr1 = get_resource_manager()
        reset_resource_manager()
        mgr2 = get_resource_manager()
        assert mgr1 is not mgr2

    def test_snapshot_cache_disk_usage(self, tmp_path):
        from hycoder.resources import CACHE_DIR
        cfg = {"max_cache_entries": "100"}
        with patch("hycoder.resources.CACHE_DIR", tmp_path / "cache"):
            mgr = ResourceManager(cfg)
            mgr.response_cache.set("test", "data")
            snap = mgr.snapshot()
            assert snap["limits"]["cache_disk_mb"] >= 0


class TestDiskCacheAdditional:
    def test_disk_usage_bytes_property(self, tmp_path):
        cache = DiskCache("test_du", max_entries=100, ttl_seconds=3600, dirpath=tmp_path)
        cache.set("a", "x")
        assert cache.disk_usage_bytes > 0

    def test_housekeeping_expired(self, tmp_path):
        cache = DiskCache("test_hk", max_entries=100, ttl_seconds=-1, dirpath=tmp_path)
        cache.set("expired", "value")
        cache._housekeeping()
        assert cache.get("expired") is None

    def test_housekeeping_lru(self, tmp_path):
        cache = DiskCache("test_lru2", max_entries=2, ttl_seconds=3600, dirpath=tmp_path)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache._housekeeping()
        assert cache.size <= 2
