"""Validated dense capture of an ALREADY RUNNING solver, no setup/teardown.

Time is the bridge's boundary label, not independently validated vendor time.
Returns full reached samples on early stop. No file writes or solver creation.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
import numpy as np
from trfc_drift import carsim_dll as cs


@dataclass(frozen=True)
class DenseSamples:
    boundary_time_s: np.ndarray
    raw_exports: np.ndarray
    interval_imports: np.ndarray
    planned_steps: int
    completed_steps: int
    solver_stopped: bool
    solver_dt_s: float


def sample_hold(plant, command, n_solver_steps, initial_exports):
    if not isinstance(n_solver_steps,int) or isinstance(n_solver_steps,bool) or not 1<=n_solver_steps<=200000:
        raise ValueError('Integer planned steps in[1,200000] required')
    dt=float(plant.dt);start=float(plant.t);n_exp=plant.n_exp
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(start) or not 1<=n_exp<=256:
        raise ValueError('Finite initial bridge time/positive dt/valid export capacity required')
    initial=np.array(initial_exports,dtype=np.float64,copy=True)
    if initial.shape!=(n_exp,) or not np.isfinite(initial).all():raise ValueError('Finite exact export vector required')
    frozen=copy.deepcopy(command)
    for name in ('steer','motor','brake','mux','muy'):
        value=np.asarray(getattr(frozen,name),dtype=np.float64)
        if value.shape!=(4,) or not np.isfinite(value).all():raise ValueError('Finite four-wheel command required:'+name)
    imports=np.empty(cs.N_IMP,dtype=np.float64);frozen.to_imports(imports)
    if not np.isfinite(imports).all():raise ValueError('Nonfinite import conversion')
    times=[start];exports=[initial];applied=[];stopped=False
    for _ in range(n_solver_steps):
        previous=float(plant.t)
        vector,stopped=plant.step(frozen,1)
        reached=float(plant.t)
        if float(plant.dt)!=dt or not math.isfinite(reached) or not math.isclose(reached-previous,dt,rel_tol=1e-9,abs_tol=1e-12):
            raise RuntimeError('Solver bridge clock/dt contract violated')
        value=np.array(vector,dtype=np.float64,copy=True)
        if value.shape!=(n_exp,) or not np.isfinite(value).all():raise RuntimeError('Invalid reached export vector')
        times.append(reached);exports.append(value);applied.append(imports.copy())
        if stopped:break
    return DenseSamples(np.array(times),np.stack(exports),np.stack(applied),n_solver_steps,len(applied),bool(stopped),dt)
