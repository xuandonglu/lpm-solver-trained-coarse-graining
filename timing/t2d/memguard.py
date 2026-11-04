"""Memory guard for my own campaign (shared machine): logs free physical memory / free commit every 20 s and, if the
free commit charge falls below LIMIT_GB, kills the process tree of the given PID (my campaign driver only).
usage: python memguard.py PID LOG [LIMIT_GB]"""
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import time


class MS(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def status():
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30, m.ullAvailPageFile / 2 ** 30


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    pid = int(sys.argv[1])
    log = open(sys.argv[2], "a")
    lim = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    t0 = time.time()
    while alive(pid):
        ph, cm = status()
        log.write("%7.0f s  free phys %5.1f GB  free commit %5.1f GB\n" % (time.time() - t0, ph, cm))
        log.flush()
        if cm < lim:
            log.write("free commit below %.1f GB -> killing my campaign tree %d\n" % (lim, pid))
            log.flush()
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            break
        time.sleep(20)
    log.write("guard exit\n")


if __name__ == "__main__":
    main()
