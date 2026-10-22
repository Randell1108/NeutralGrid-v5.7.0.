"""Read-only process identity for WebSocket ownership and shutdown checks."""

from __future__ import annotations

import ctypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProcessObservation:
    pid: int
    state: str
    creation_token: str | None = None
    executable: str | None = None

    def identity(self) -> dict[str, Any] | None:
        if self.state != "running" or not self.creation_token or not self.executable:
            return None
        return {key: value for key, value in asdict(self).items() if key != "state"}


def query_process(pid: int) -> ProcessObservation:
    """Return running/exited/unknown; query failure alone never establishes death."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return ProcessObservation(pid, "unknown")
    if os.name == "nt":
        return _query_windows(pid)
    if Path("/proc").is_dir():
        return _query_procfs(pid)
    return ProcessObservation(pid, "unknown")


def _windows_pid_absent(kernel32: Any, pid: int) -> bool:
    """Require two complete native inventories; inaccessible present PIDs stay unknown."""
    from ctypes import wintypes

    try:
        enumerate_processes = kernel32.K32EnumProcesses
        enumerate_processes.argtypes = (
            ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
        )
        enumerate_processes.restype = wintypes.BOOL
        for _ in range(2):
            capacity = 1024
            while capacity <= 65536:
                buffer = (wintypes.DWORD * capacity)()
                size = ctypes.sizeof(buffer)
                returned = wintypes.DWORD()
                if not enumerate_processes(buffer, size, ctypes.byref(returned)):
                    return False
                count_bytes = returned.value
                if count_bytes > size or count_bytes % ctypes.sizeof(wintypes.DWORD):
                    return False
                if count_bytes == size:
                    # EnumProcesses does not otherwise report buffer truncation.
                    capacity *= 2
                    continue
                pids = buffer[:count_bytes // ctypes.sizeof(wintypes.DWORD)]
                if os.getpid() not in pids or pid in pids:
                    return False
                break
            else:
                return False
        return True
    except (AttributeError, OSError, ValueError):
        return False


def _query_windows(pid: int) -> ProcessObservation:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        absent = error == 87 or (error == 5 and _windows_pid_absent(kernel32, pid))
        return ProcessObservation(pid, "exited" if absent else "unknown")
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return ProcessObservation(pid, "unknown")
        if exit_code.value != 259:
            return ProcessObservation(pid, "exited")
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            return ProcessObservation(pid, "unknown")
        size = wintypes.DWORD(32768)
        executable = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, executable, ctypes.byref(size)):
            return ProcessObservation(pid, "unknown")
        token = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return ProcessObservation(pid, "running", f"win32_filetime:{token}", executable.value)
    finally:
        kernel32.CloseHandle(handle)


def _query_procfs(pid: int) -> ProcessObservation:
    process_dir = Path("/proc") / str(pid)
    try:
        stat = (process_dir / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return ProcessObservation(pid, "exited")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        executable = str((process_dir / "exe").readlink())
        # Field 22 is starttime; fields starts at field 3 (state).
        token = f"procfs:{boot_id}:{fields[19]}"
        return ProcessObservation(pid, "running", token, executable)
    except FileNotFoundError:
        return ProcessObservation(pid, "unknown" if process_dir.exists() else "exited")
    except (OSError, UnicodeError, IndexError):
        return ProcessObservation(pid, "unknown")


def matches_identity(observed: ProcessObservation, expected: Any) -> bool:
    """Match an immutable OS creation token and executable, never PID alone."""
    return (
        observed.state == "running"
        and isinstance(expected, Mapping)
        and isinstance(expected.get("pid"), int)
        and not isinstance(expected.get("pid"), bool)
        and observed.pid == expected.get("pid")
        and bool(observed.creation_token)
        and observed.creation_token == expected.get("creation_token")
        and isinstance(expected.get("executable"), str)
        and bool(expected.get("executable"))
        and bool(observed.executable)
        and os.path.normcase(str(observed.executable)) == os.path.normcase(expected["executable"])
    )
