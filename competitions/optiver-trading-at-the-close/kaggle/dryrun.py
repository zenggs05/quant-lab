"""Full dry run of the built notebook under the local API replay, measuring wall time and peak RSS.

Usage (from kaggle/build, Python 3.10 env):
  OPTIVER_* env vars ... ../../.venv-kaggle310/python.exe -X utf8 ../dryrun.py submission_model.py
"""
import os
import runpy
import sys
import threading
import time

import psutil

peak = [0]
stop = [False]


def watch():
    p = psutil.Process()
    while not stop[0]:
        peak[0] = max(peak[0], p.memory_info().rss)
        time.sleep(0.5)


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_api"))
t = threading.Thread(target=watch, daemon=True)
t.start()
t0 = time.time()
runpy.run_path(sys.argv[1], run_name="__main__")
stop[0] = True
print(f"DRYRUN wall={time.time()-t0:.0f}s peak_rss={peak[0]/2**30:.2f} GiB")
