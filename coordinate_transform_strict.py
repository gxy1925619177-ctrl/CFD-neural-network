#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math, numpy as np
try:
    from scipy.optimize import fsolve
except Exception:
    fsolve = None

# constants (match MATLAB)
r = 0.01905
D = 2 * r
straight_length = 1.98891
axis_distance = 0.4
bend_radius = 0.03014
bend_angle = math.pi/2
bend_arc_length = bend_radius * bend_angle
bottom_length = axis_distance - 2*bend_radius
x_import_straight_center = 0.2
x_export_straight_center = -0.2
z_bend1_center = straight_length
x_bend1_center = 0.16986
z_bend2_center = straight_length
x_bend2_center = -0.16986
x_horizontal_range = (-0.16986, 0.16986)
z_horizontal_center = 2.01915

L_import_straight_end = straight_length
L_first_bend_end = L_import_straight_end + bend_arc_length
L_bottom_end = L_first_bend_end + bottom_length
L_second_bend_end = L_bottom_end + bend_arc_length
L_export_straight_end = L_second_bend_end + straight_length
TOTAL_L = L_export_straight_end

def create_radial_plane_basis(normal):
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if abs(normal[0]) < abs(normal[1]) and abs(normal[0]) < abs(normal[2]):
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, v); u = u / (np.linalg.norm(u)+1e-12)
    v = np.cross(normal, u); v = v / (np.linalg.norm(v)+1e-12)
    return u, v

def _solve_center(fun, x0, z0):
    if fsolve is not None:
        try:
            sol = fsolve(lambda vv: fun(vv[0], vv[1]), [x0, z0], maxfev=1000)
            return float(sol[0]), float(sol[1])
        except Exception:
            pass
    return float(x0), float(z0)

def xyz_to_Ltheta_strict(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    N = x.size
    L = np.full(N, np.nan, dtype=float)
    theta = np.full(N, np.nan, dtype=float)
    segment = np.zeros(N, dtype=int)

    # 1) import straight
    mask = (z <= straight_length + 1e-9) & (x > 0)
    if np.any(mask):
        xi, yi, zi = x[mask], y[mask], z[mask]
        L[mask] = zi
        dx = xi - x_import_straight_center
        dy = yi - 0.0
        theta[mask] = (np.degrees(np.arctan2(dy, dx)) % 360.0)
        segment[mask] = 1

    # 2) export straight
    mask = (z <= straight_length + 1e-9) & (x < 0)
    if np.any(mask):
        xi, yi, zi = x[mask], y[mask], z[mask]
        L[mask] = L_second_bend_end + straight_length - zi
        dx = xi - x_export_straight_center
        dy = yi - 0.0
        theta[mask] = (np.degrees(np.arctan2(dy, -dx)) % 360.0)
        segment[mask] = 2

    # 3) bottom horizontal (y-z plane)
    mask = (x >= x_horizontal_range[0]) & (x <= x_horizontal_range[1]) & (segment == 0)
    if np.any(mask):
        xi, yi, zi = x[mask], y[mask], z[mask]
        s_rel = (x_horizontal_range[1] - xi) / (x_horizontal_range[1] - x_horizontal_range[0])
        L[mask] = L_first_bend_end + s_rel * bottom_length
        dz = zi - z_horizontal_center
        dy = yi - 0.0
        theta[mask] = (np.degrees(np.arctan2(dy, dz)) % 360.0)
        segment[mask] = 3

    # 4) bend1 (import side)
    mask = (z > straight_length - 1e-9) & (x >= -1e-9) & (segment == 0)
    if np.any(mask):
        idx = np.where(mask)[0]
        for i in idx:
            xi, yi, zi = x[i], y[i], z[i]
            cx, cz = x_bend1_center, z_bend1_center
            dxg = xi - cx; dzg = zi - cz
            g = max(0.0, min(math.pi/2.0, math.atan2(dzg, dxg)))
            x0g = cx + bend_radius * math.cos(g)
            z0g = cz + bend_radius * math.sin(g)
            def fun(x0v, z0v):
                return [-(z0v - cz)*(x0v - xi) + (x0v - cx)*(z0v - zi),
                        (x0v - cx)**2 + (z0v - cz)**2 - bend_radius**2]
            x0i, z0i = _solve_center(fun, x0g, z0g)
            ang = math.atan2(z0i - cz, x0i - cx)
            if not (0.0 <= ang <= math.pi/2.0):
                x0i = 2*cx - x0i; z0i = 2*cz - z0i
                ang = math.atan2(z0i - cz, x0i - cx)
            L[i] = L_import_straight_end + ang * bend_radius
            dxr, dyr, dzr = xi - x0i, yi - 0.0, zi - z0i
            tangent = np.array([-(z0i - cz), 0.0, (x0i - cx)], float)
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            u, v = create_radial_plane_basis(normal)
            pu = - (dxr*u[0] + dyr*u[1] + dzr*u[2]); pv = - (dxr*v[0] + dyr*v[1] + dzr*v[2])
            theta[i] = (math.degrees(math.atan2(pv, pu)) % 360.0); segment[i] = 4

    # 5) bend2 (export side)
    mask = (z > straight_length) & (x < 0) & (segment == 0)
    if np.any(mask):
        idx = np.where(mask)[0]
        for i in idx:
            xi, yi, zi = x[i], y[i], z[i]
            cx, cz = x_bend2_center, z_bend2_center
            dxg = cx - xi; dzg = zi - cz
            g = max(0.0, min(math.pi/2.0, math.atan2(dzg, dxg)))
            x0g = cx - bend_radius * math.cos(g)
            z0g = cz + bend_radius * math.sin(g)
            def fun(x0v, z0v):
                return [-(z0v - cz)*(x0v - xi) + (x0v - cx)*(z0v - zi),
                        (x0v - cx)**2 + (z0v - cz)**2 - bend_radius**2]
            x0i, z0i = _solve_center(fun, x0g, z0g)
            ang = math.atan2(z0i - cz, cx - x0i)
            if not (0.0 <= ang <= math.pi/2.0):
                x0i = 2*cx - x0i; z0i = 2*cz - z0i
                ang = math.atan2(z0i - cz, cx - x0i)
            L[i] = L_bottom_end + ang * bend_radius
            dxr, dyr, dzr = xi - x0i, yi - 0.0, zi - z0i
            tangent = np.array([(z0i - cz), 0.0, (cx - x0i)], float)
            normal = tangent / (np.linalg.norm(tangent) + 1e-12)
            u, v = create_radial_plane_basis(normal)
            pu = (dxr*u[0] + dyr*u[1] + dzr*u[2]); pv = (dxr*v[0] + dyr*v[1] + dzr*v[2])
            theta[i] = (math.degrees(math.atan2(pv, pu)) % 360.0); segment[i] = 5

    return {"L": L, "theta": theta, "segment": segment, "L_total": TOTAL_L}
