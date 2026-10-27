"""Explicit candidate Euler convention for offline coordinate closure.

Rz(yaw)Ry(pitch)Rx(roll), body-to-world. Not a vendor-frame assumption or a
complete generalized mass/force model; must pass actual coordinate checks.
"""
import torch


def euler_rotation(angles):
    if angles.shape[-1]!=3 or not torch.is_floating_point(angles) or not bool(torch.isfinite(angles).all()):
        raise ValueError('Finite floating yaw/pitch/roll in radians required')
    yaw,pitch,roll=angles.unbind(-1);cy,cp,cr=map(torch.cos,(yaw,pitch,roll));sy,sp,sr=map(torch.sin,(yaw,pitch,roll))
    return torch.stack((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,
                        sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,
                        -sp,cp*sr,cp*cr),-1).reshape(angles.shape[:-1]+(3,3))


def wheel_local(wheel_world,cg_world,angles,cg_local):
    if wheel_world.shape[-2:]!=(4,3) or cg_world.shape[-1]!=3 or cg_local.shape!=(3,):raise ValueError('Four XYZ wheels and explicit CG coordinates required')
    if not all(bool(torch.isfinite(x).all()) for x in (wheel_world,cg_world,cg_local)):raise ValueError('Finite positions required')
    R=euler_rotation(angles)
    return torch.einsum('...ij,...wj->...wi',R.transpose(-1,-2),wheel_world-cg_world.unsqueeze(-2))+cg_local
