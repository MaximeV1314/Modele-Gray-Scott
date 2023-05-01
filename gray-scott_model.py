import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
import glob
import pandas as pd
import random as rd
from matplotlib.colors import LogNorm

mpl.use('Agg')

################################################
######## Initialisation des variables ##########
################################################

dx = 1
dy = 1
Lx = 400
Ly = 400

X = np.arange(0, Lx+dx, dx)
Y = np.arange(0, Ly+dy, dy)

x, y = np.meshgrid(X, Y)

dt = 1
tf = 10000
t = np.arange(0, tf+dt, dt)
pas_img = 30
digit = 4

Du = 0.1
Dv = 0.05
F = 0.0500
K = 0.0650

F = np.linspace(0.03, 0.05, (Lx-2) * (Ly-2)).reshape(Lx-2, Ly-2)  #0.03, 0.05
K = np.linspace(0.055, 0.065, (Lx-2) * (Ly-2)).reshape(Lx-2, Ly-2)  #0.055, 0.065

################################################
################## Fonctions ###################
################################################

def name(i,digit):
    """Fonction nommant les images dans le fichier img"""

    i = str(i)
    while len(i)<digit:
        i = '0'+i
    i = 'img/'+i+'.png'

    return(i)
############ fonctions onde initiale ###############

def random():

    d = 2

    X_v = np.random.randint(d, Lx-d, Nv)
    Y_v = np.random.randint(d, Ly-d, Nv)

    u = np.ones((Lx,Ly))
    v = np.zeros((Lx,Ly))

    for i in range(Nv):

        u[X_v[i] - d : X_v[i] + d, Y_v[i] - d : Y_v[i] + d] = np.ones((2*d, 2*d)) * 0.5
        v[X_v[i] - d : X_v[i] + d, Y_v[i] - d : Y_v[i] + d] = np.ones((2*d, 2*d)) * 0.25

    return u, v

def init(n):

    u = np.ones((n+2,n+2))
    v = np.zeros((n+2,n+2))

    x, y = np.meshgrid(np.linspace(0, 1, n+2), np.linspace(0, 1, n+2))

    mask = (0.4<x) & (x<0.6) & (0.4<y) & (y<0.6)

    u[mask] = 0.50
    v[mask] = 0.25

    return u, v

############ fonctions de bords ###############

def tor_x(u, v):
    u[:, 0] += u[:, -2]
    u[:, -1] += u[:, 1]
    v[:, 0] += v[:, -2]
    v[:, -1] += v[:, 1]
    return u, v

def tor_y(u, v):
    u[0, :] += u[-2, :]
    u[-1, :] += u[1, :]
    v[0, :] += v[-2, :]
    v[-1, :] += v[1, :]

    return u, v

############ update ###############

def update(u0, v0):

    u = np.zeros((Lx, Ly))
    v = np.zeros((Lx, Ly))

    u[1:Lx-1, 1:Ly-1] = Du * dt/dx**2 * (u0[0:Lx-2, 1:Ly-1] + u0[2:Lx, 1:Ly-1]) + \
                        Du * dt/dy**2 * (u0[1:Lx-1, 0:Ly-2] + u0[1:Lx-1, 2:Ly]) + \
                        u0[1:Lx-1, 1:Ly-1] * (1 - dt * v0[1:Lx-1, 1:Ly-1]**2 - 2 * Du * dt * (1/dx**2 + 1/dy**2) - dt * F) + dt * F

    v[1:Lx-1, 1:Ly-1] = Dv * dt/dx**2 * (v0[0:Lx-2, 1:Ly-1] + v0[2:Lx, 1:Ly-1]) + \
                        Dv * dt/dy**2 * (v0[1:Lx-1, 0:Ly-2] + v0[1:Lx-1, 2:Ly]) + \
                        v0[1:Lx-1, 1:Ly-1] * (1 + dt * v0[1:Lx-1, 1:Ly-1] * u0[1:Lx-1, 1:Ly-1] - 2 * Dv * dt * (1/dx**2 + 1/dy**2) - dt * (F + K))

    u, v = tor_x(u, v)
    u, v = tor_y(u, v)

    return u, v

################################################
############### Ondes initiales ################
################################################

Nv = 100

u0, v0 = random()
#u0, v0 = init(Lx-2)

u_arr = [u0]
v_arr = [v0]

################################################
################## Animation ###################
################################################

for i in range(len(t)):
    u0, v0 = update(u0, v0)
    if i%pas_img == 0:
        u_arr.append(u0)
        v_arr.append(v0)

A_max = np.max(u_arr)
A_min = np.min(u_arr)

extension="img/*.png"
for f in glob.glob(extension):
  os.remove(f)

for i in range(0, len(u_arr)-1):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.axis('equal')

    im = ax.imshow(u_arr[i], vmin = 0, vmax = 1, cmap = "viridis_r", interpolation = "gaussian")
    #im = ax.imshow(psi_arr[i], extent = (-Ly, Ly, -Lx, Lx), cmap = "summer", interpolation = "gaussian")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax)
    #cbar = fig.colorbar(im, ax=ax, fraction=0.0323, pad=0.04)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)

    name_pic = name(int(i), digit)
    plt.savefig(name_pic, bbox_inches='tight', dpi=300)

    ax.clear()
    plt.close(fig)

    print(i/len(u_arr))

print("Images successed")

  #ffmpeg -r 10 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    #ffmpeg -r 30 -i img/%04d.png -vcodec libx264 -y -an test.mp4 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"