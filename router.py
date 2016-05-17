# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:39:28 2016

@author: petteri
"""

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

# spherical coordinates:
# (altitude, latitude, lognitude)
# (r, phi, theta)
# (0 ... 700, -90 ... 90, -180 ... 180)

R_earth = 6371. # kilometers

def spherical2cart(p_deg):
    p = np.zeros(p_deg.shape)
    p[0] = p_deg[0]
    p[1] = np.deg2rad(p_deg[1])
    p[2] = np.deg2rad(p_deg[2])

    r = R_earth + p[0]
    x = r * np.cos(p[1]) * np.cos(p[2])
    y = r * np.cos(p[1]) * np.sin(p[2])
    z = r * np.sin(p[1])
    return np.array([x,y,z])

def distance(p):
    return np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

def is_visible(s1, s2):
    """Parametrizes a line between the two points (given in spherical
    coordinates) and checks whether the line hits the earth (i.e. elevation
    of the line is always above earth radius)"""

    t = np.linspace(1e-4,1,9999,endpoint=False)
    p1 = spherical2cart(s1)
    p2 = spherical2cart(s2)
    los = p1 + np.outer(t, (p2 - p1))
    dist = np.zeros(los.shape[0])
    for i in np.arange(len(dist)):
        dist[i] = distance(los[i,:])

    return np.all(dist >= R_earth)

# count satellites
satellite_count = 0
with open('data.csv', 'r') as datafile:
    for line in datafile:
        if line.startswith('SAT'):
            satellite_count += 1

satellites = np.zeros((satellite_count + 2, 3))
satellite_ids = ['start']

# read satellite positions. Treat start and
# end points as satellites with elevation = 0

sat_i = 1
with open('data.csv', 'r') as datafile:
    for line in datafile:
        if line.startswith('SAT'):
            fields = line.split(',')
            satellite_ids.append(fields[0])
            satellites[sat_i,0] = np.double(fields[3])
            satellites[sat_i,1] = np.double(fields[1])
            satellites[sat_i,2] = np.double(fields[2])
            sat_i += 1
        elif line.startswith('ROUTE'):
            fields = line.split(',')
            satellites[0,1] = np.double(fields[1])
            satellites[0,2] = np.double(fields[2])
            satellites[-1,1] = np.double(fields[3])
            satellites[-1,2] = np.double(fields[4])
satellite_ids.append('end')

end_idx = len(satellite_ids) - 1
routes = [(0,)]
route_found = np.all(satellites[0,:] == satellites[-1,:]) # just in case the start- and endpoints are the same :)
satellites_seen = [0]
while (not route_found):
    next_fwd = []
    new_routes = []
    for route in routes:
        print route
        s1i = route[-1]
        for (s2i, s2) in enumerate(satellites):
            if is_visible(satellites[s1i,:], s2) and s2i not in satellites_seen:
                new_routes.append(route + (s2i,))
                satellites_seen.append(s2i)
                if s2i == end_idx:
                    route_found = True
    routes = new_routes
    if len(routes) == 0: # we've seen all satellites but haven't found the endpoint
        print 'empty'
        break
    print routes

# In case we found two paths with the same number of hops,
# choose the shorter one
shortest_dist = np.inf
shortest = routes[0]
for route in routes:
    if route[-1] == end_idx:
        dist = 0.0
        for i in np.arange(1,len(route)):
            dist += distance(satellites[i-1] - satellites[i])
        if dist < shortest_dist:
            shortest = route


# plot the result
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# plot the globe
x = R_earth * np.outer(np.cos(u), np.sin(v))
y = R_earth * np.outer(np.sin(u), np.sin(v))
z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='0.8', alpha=0.2)

# plot all satellites
for sat in satellites[1:-1]:
    p = spherical2cart(sat)
    ax.scatter(p[0],p[1],p[2], alpha=0.2)

# A sanity check
#p = spherical2cart(np.array([0,0,0]))
#ax.scatter(p[0],p[1],p[2], c='black', marker='o')
#p = spherical2cart(np.array([0,90,0]))
#ax.scatter(p[0],p[1],p[2], c='black', marker='o')

# the starting point
p = spherical2cart(satellites[0,:])
ax.scatter(p[0],p[1],p[2], c='r')

# the intermediate points
for idx in np.arange(1,len(shortest)):
    p = spherical2cart(satellites[shortest[idx],:])
    ax.scatter(p[0],p[1],p[2])

    t = np.linspace(1e-4,1,9999)
    p1 = spherical2cart(satellites[shortest[idx-1],:])
    p2 = spherical2cart(satellites[shortest[idx],:])
    los = p1 + np.outer(t, (p2 - p1))
    ax.plot(los[:,0], los[:,1], los[:,2])
# the endpoint
p = spherical2cart(satellites[-1,:])
ax.scatter(p[0],p[1],p[2], c='g')


# Show the answer
answer = ''
for sat_idx in shortest[1:-1]:
    answer += satellite_ids[sat_idx] + ','
answer = answer[:-1]
print answer
