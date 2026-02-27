import numpy as np

from ecl_analysis.cache import FrameCache


def test_frame_cache_put_get_and_copy():
    cache = FrameCache(max_size=2)
    frame = np.arange(4, dtype=np.uint8).reshape(2, 2)

    cache.put(0, frame)
    retrieved = cache.get(0)

    assert retrieved is not None
    assert np.array_equal(retrieved, frame)

    # Ensure a defensive copy is returned
    retrieved[0, 0] = 99
    cached_again = cache.get(0)
    assert cached_again[0, 0] != 99


def test_frame_cache_eviction_order():
    cache = FrameCache(max_size=2)
    cache.put(0, np.zeros((1, 1), dtype=np.uint8))
    cache.put(1, np.ones((1, 1), dtype=np.uint8))
    # Access frame 0 to mark it as most recently used
    _ = cache.get(0)

    cache.put(2, np.full((1, 1), 2, dtype=np.uint8))

    assert cache.get(1) is None, "Least recently used frame should be evicted"
    assert cache.get(0) is not None
    assert cache.get(2) is not None


def test_frame_cache_clear():
    cache = FrameCache(max_size=2)
    cache.put(0, np.zeros((1, 1), dtype=np.uint8))
    cache.put(1, np.zeros((1, 1), dtype=np.uint8))
    assert cache.get_size() == 2

    cache.clear()

    assert cache.get_size() == 0
    assert cache.get(0) is None
