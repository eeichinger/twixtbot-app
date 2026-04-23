#! /usr/bin/env python
import argparse
from collections import namedtuple
import datetime
import os
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num_clones", "-n", type=int, required=True)
parser.add_argument("--log_dir", "-l", type=str, required=True)
parser.add_argument("cmdline", nargs='+')
args = parser.parse_args()

os.mkdir(args.log_dir)
master_log = os.path.join(args.log_dir, "master.log")
print("logging to %s" % (master_log))
log_f = open(master_log, 'w')
sys.stdout = log_f
sys.stderr = log_f

def mini_log10(n):
    assert n > 0
    k = 1
    while n >= 10:
        n //= 10
        k += 1
    return k

def when():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d %H:%M:%S")

def search_replace_cmd(org_cmdline, name):
    return [x.replace("%n%", name) for x in org_cmdline]

num_digits = mini_log10(max(1,args.num_clones - 1))
sys.stdin.close()
os.close(0)

ProcInfo = namedtuple('ProcInfo', ['p', 'id', 'fileobj'])
procmap = dict()

_start_time = time.time()
_last_progress = _start_time
_PROGRESS_INTERVAL = 60
_total = args.num_clones
_completed = 0

for i in range(args.num_clones):
    print(when(), "start instance #%d" % i)
    sys.stdout.flush()
    name = "%0*d" % (num_digits, i)
    outfile = os.path.join(args.log_dir, name + ".log")
    f = open(outfile, 'w')
    cmd = search_replace_cmd(args.cmdline, name)
    p = subprocess.Popen(cmd, stdout=f, stderr=f)
    proc = ProcInfo(p, i, f)
    procmap[p.pid] = proc

_failures = 0
while procmap:
    (pid, status) = os.wait()
    _completed += 1
    proc = procmap[pid]
    signum = status & 0xff
    xcode = (status >> 8) & 0xff
    if signum:
        print(when(), "instance %d exited with signal %d" % (proc.id, signum))
        _failures += 1
    elif xcode:
        print(when(), "instance %d exited with status %d" % (proc.id, xcode))
        _failures += 1
    else:
        print(when(), "instance %d finished happily" % (proc.id))
    proc.p.wait()
    proc.fileobj.close()
    del procmap[pid]

    now = time.time()
    elapsed = now - _start_time
    if now - _last_progress >= _PROGRESS_INTERVAL:
        remaining_ids = " ".join(["%0*d" % (num_digits, p.id)
                                  for p in sorted(procmap.values(), key=lambda x: x.id)])
        rate = _completed / elapsed if elapsed > 0 else 0
        eta = len(procmap) / rate if rate > 0 else 0
        print(when(), "progress %d/%d done | elapsed %.0fs | ETA ~%.0fs | remaining: %s" % (
            _completed, _total, elapsed, eta, remaining_ids or "none"))
        _last_progress = now

    sys.stdout.flush()

elapsed_total = time.time() - _start_time
print(when(), "all instances finished. Total: %.1fs (%.1f min)" % (elapsed_total, elapsed_total / 60))
if _failures:
    print(when(), "FAIL: %d of %d instances exited with error" % (_failures, _total))
    log_f.close()
    sys.exit(1)
log_f.close()
