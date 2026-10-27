"""Offline timing gate for conventional full-step native solver captures.

No solver creation, unit guessing, resampling, file writes or channel shifts.
Metadata must come from the actual run, not assumed vendor defaults.
"""
from __future__ import annotations
import hashlib
import numpy as np
from trfc_drift.dense_solver_sampler150_v1 import DenseSamples


def capture_identity(data):
    h=hashlib.sha256()
    for a in (data.boundary_time_s,data.raw_exports,data.interval_imports):
        v=np.ascontiguousarray(a)
        h.update(str((v.shape,v.dtype.str)).encode('ascii'));h.update(v.tobytes())
    h.update(str((data.planned_steps,data.completed_steps,data.solver_stopped,data.solver_dt_s)).encode('ascii'))
    return h.hexdigest()


def audit_timing(data: DenseSamples, stamp_index: int, *, integration_method,
                 io_sync_fm, actual_run_metadata_sha256: str, atol_s=1e-9):
    if integration_method not in (-1, 0) or isinstance(integration_method, bool):
        raise ValueError('Resolved full-step Euler/AB-2 mode required')
    if io_sync_fm != 0 or isinstance(io_sync_fm, bool):
        raise ValueError('Conventional common-time exports required; mixed stages unsupported')
    if (not isinstance(actual_run_metadata_sha256, str) or len(actual_run_metadata_sha256)!=64
        or any(c not in '0123456789abcdef' for c in actual_run_metadata_sha256)):
        raise ValueError('Actual-run metadata SHA256 required')
    if not isinstance(stamp_index, int) or isinstance(stamp_index, bool):
        raise ValueError('Explicit T_Stamp column required')
    b=np.asarray(data.boundary_time_s, dtype=np.float64)
    e=np.asarray(data.raw_exports, dtype=np.float64)
    u=np.asarray(data.interval_imports, dtype=np.float64)
    n=data.completed_steps
    if (not isinstance(n, int) or isinstance(n, bool) or n<1 or n>data.planned_steps
        or b.shape!=(n+1,) or e.ndim!=2 or e.shape[0]!=n+1
        or u.shape!=(n,24) or not 0<=stamp_index<e.shape[1]):
        raise ValueError('Capture dimensions/counts invalid')
    if not (np.isfinite(b).all() and np.isfinite(e).all() and np.isfinite(u).all()
            and np.isfinite(data.solver_dt_s) and data.solver_dt_s>0
            and np.isfinite(atol_s) and 0<atol_s<=1e-8):
        raise ValueError('Finite data and bounded positive time tolerance required')
    if not np.allclose(np.diff(b),data.solver_dt_s,rtol=1e-9,atol=1e-12):
        raise ValueError('Bridge advancement inconsistent')
    api_time=b[:-1].copy()
    stamp=e[:,stamp_index].copy()
    # Initial copy_io and the first conventional integrate_io both represent t0.
    expected=np.concatenate((b[:1],api_time))
    difference=stamp-expected
    return dict(pass_=bool(np.max(np.abs(difference))<=atol_s),
                capture_sha256=capture_identity(data),
                actual_run_metadata_sha256=actual_run_metadata_sha256,
                api_call_time_s=api_time, bridge_after_call_time_s=b[1:].copy(),
                export_calculation_time_s=stamp, expected_export_time_s=expected,
                max_clock_error_s=float(np.max(np.abs(difference))),
                planned_steps=data.planned_steps, completed_steps=n,
                solver_stopped=data.solver_stopped, solver_dt_s=data.solver_dt_s,
                initial_row_excluded_from_regular_grid=True)


def calculation_grid_view(data: DenseSamples, audit: dict, stride: int):
    """Select returned calculation rows; initial copy_io stays in the raw record."""
    if audit.get('pass_') is not True:
        raise ValueError('Clock gate must pass before decimation')
    if not isinstance(stride,int) or isinstance(stride,bool) or stride<1:
        raise ValueError('Positive integer stride required')
    # Do not trust a foreign/stale audit for a different capture.
    if (audit.get('capture_sha256')!=capture_identity(data)
        or audit['completed_steps']!=data.completed_steps or audit['solver_dt_s']!=data.solver_dt_s
        or not np.array_equal(audit['expected_export_time_s'],np.r_[data.boundary_time_s[0],data.boundary_time_s[:-1]])):
        raise ValueError('Audit/capture identity mismatch')
    indices=np.arange(1,data.completed_steps+1,stride,dtype=np.int64)
    stamps=audit['export_calculation_time_s']
    if len(stamps)!=len(data.raw_exports):raise ValueError('Audit row mismatch')
    return dict(raw_row_indices=indices, time_s=stamps[indices].copy(),
                raw_exports=data.raw_exports[indices].copy(),
                discarded_initial_rows=1,
                unretained_tail_steps=int(data.completed_steps-indices[-1]),
                covered_duration_s=float(stamps[indices[-1]]-stamps[indices[0]]),
                planned_steps=data.planned_steps, completed_steps=data.completed_steps)
