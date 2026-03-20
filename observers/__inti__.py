"""

Observer pattern package.

Re-exports the public interface so callers can write:
    from observers import BaseObserver
instead of the full path.
"""

from observers.base_observer import BaseObserver, ObserverError

__all__ = ["BaseObserver", "ObserverError"]
