"""Shared exclusive lock for MedAgent workers; never kills or pauses foreign jobs."""
import fcntl
import subprocess
import time
from contextlib import contextmanager
from .common import ROOT,read_json


def allowed_indices():
    path=ROOT/'configs/runtime.json'
    configured=read_json(path).get('allowed_gpu_indices',[0,1]) if path.exists() else [0,1]
    if not configured or any(type(i) is not int or i not in (0,1,2,3) for i in configured):
        raise ValueError('Configured GPU pool must be within physical GPUs 0–3')
    return set(configured)


def snapshot():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    busy=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    occupied=set(busy.stdout.splitlines())
    return [{'index':int(a),'uuid':b.strip(),'free_mib':int(c),'utilization':int(d),'has_process':b.strip() in occupied}
            for a,b,c,d in [line.split(',') for line in p.stdout.strip().splitlines()]]


def idle_gpu():
    allowed=allowed_indices()
    return next((g['index'] for g in snapshot() if g['index'] in allowed and not g['has_process'] and g['free_mib']>=28000 and g['utilization']<10),None)


@contextmanager
def exclusive():
    p=ROOT/'runtime/gpu.lock';p.parent.mkdir(exist_ok=True,mode=0o700)
    with p.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


@contextmanager
def reserve_idle(on_wait):
    while True:
        if idle_gpu() is None:
            on_wait();time.sleep(30);continue
        with exclusive():
            gpu=idle_gpu()
            if gpu is None: continue
            yield gpu
            return


@contextmanager
def reserve_specific(index,on_wait):
    if index not in allowed_indices():raise ValueError('GPU index is outside the configured pool')
    path=ROOT/'runtime'/f'gpu-{index}.lock'
    with path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            while True:
                g=next((g for g in snapshot() if g['index']==index),None)
                if g is None:raise RuntimeError('configured GPU not found')
                if not g['has_process'] and g['free_mib']>=28000 and g['utilization']<10:break
                on_wait();time.sleep(30)
            yield index
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


@contextmanager
def reserve_many(indices,on_wait):
    indices=list(indices)
    if not indices or len(set(indices))!=len(indices) or not set(indices)<=allowed_indices():
        raise ValueError('invalid_gpu_reservation')
    while True:
        locks=[];available=False
        try:
            try:
                for index in sorted(indices):
                    lock=(ROOT/'runtime'/f'gpu-{index}.lock').open('a')
                    locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                states={s['index']:s for s in snapshot()}
                available=all(i in states and not states[i]['has_process'] and states[i]['free_mib']>=40000 for i in indices)
            except BlockingIOError:available=False
            if available:
                yield indices
                return
        finally:
            for lock in reversed(locks):lock.close()
        on_wait();time.sleep(20)


@contextmanager
def try_training_pair():
    """Nonblocking pair reservation lets the queue service other runnable jobs."""
    pairs=[pair for pair in ([0,1],[2,3]) if set(pair)<=allowed_indices()]
    for pair in pairs:
        locks=[];available=False
        try:
            try:
                for index in pair:
                    lock=(ROOT/'runtime'/f'gpu-{index}.lock').open('a')
                    locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                states={s['index']:s for s in snapshot()}
                available=all(i in states and not states[i]['has_process'] and states[i]['free_mib']>=40000 for i in pair)
            except BlockingIOError:available=False
            if available:
                yield pair
                return
        finally:
            for lock in reversed(locks):lock.close()
    yield None


@contextmanager
def reserve_training_pair(on_wait):
    if not any(set(pair)<=allowed_indices() for pair in ([0,1],[2,3])):
        raise ValueError('no_allowed_training_pair')
    while True:
        with try_training_pair() as pair:
            if pair is not None:
                yield pair
                return
        on_wait();time.sleep(20)
