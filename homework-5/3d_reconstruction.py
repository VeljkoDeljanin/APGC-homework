import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p1l = [1195, 605]
p2l = [867, 1039]
p3l = [512, 782]
p4l = [871, 452]
p5l = [1235, 446]
p6l = [874, 896]
p7l = [462, 622]
p8l = [875, 306]
p9l = [1136, 519]
p10l = [708, 705]
p11l = [599, 547]
p12l = [998, 398]
p13l = [1152, 424]
p14l = [700, 607]
p15l = [589, 463]
p16l = [1009, 305]

p1r = [822, 908]
p2r = [447, 574]
p3r = [692, 381]
p4r = [1058, 652]
p5r = [840, 819]
p6r = [406, 448]
p7r = [690, 252]
p8r = [1109, 534]
p9r = [729, 685]
p10r = [572, 373]
p11r = [745, 319]
p12r = [925, 617]
p13r = [732, 620]
p14r = [566, 295]
p15r = [742, 245]
p16r = [934, 543]

left = [p1l, p2l, p3l, p4l, p5l, p6l, p7l, p8l, p9l, p10l, p11l, p12l, p13l, p14l, p15l, p16l]
right = [p1r, p2r, p3r, p4r, p5r, p6r, p7r, p8r, p9r, p10r, p11r, p12r, p13r, p14r, p15r, p16r]

# image: 1600 x 1200

def fix_coords(point):
    return [1600 - point[0], point[1], 1]

left_fixed = list(map(fix_coords, left))
right_fixed = list(map(fix_coords, right))

# Odredjivanje fundamentalne matrice F
def equation(left_point, right_point):
    return [a * b for a in left_point for b in right_point]

matrix_form = [equation(left, right) for left, right in zip(left_fixed, right_fixed)]
U, D, V = LA.svd(matrix_form)
F = np.array(V[-1])
FF = F.reshape(3, 3).T
print('Fundamentalna matrica FF: \n', FF)
print()

# Odredjivanje epipolova
UU, DD, VV = LA.svd(FF)
e1 = np.array(VV[-1])
e1 = e1 * (1 / e1[-1])

e2 = np.array(UU.T[-1])
e2 = e2 * (1 / e2[-1])

# "Popravka" fundamentalne matrice
DD = np.diag(DD)
DD1 = np.diag([1, 1, 0])
DD1 = DD1 @ DD
FF1 = (UU @ DD1) @ VV
print('"Popravka" fundamentalne matrice FF1: \n', FF1)
print()

# Odredjivanje osnovne matrice E
K1 = np.array([
    [1300,    0, 800],
    [   0, 1300, 600],
    [   0,    0,   1]
])
K2 = K1

EE = (K2.T @ FF1) @ K1
print('Osnovna matrica EE: \n', EE)
print()

# Dekompozicija osnovne matrice E
Q0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
E0 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

U, SS, V = LA.svd(-EE)

EC = (U @ E0) @ U.T
AA = (U @ Q0.T) @ V

print('Matrica EC: \n', EC)
print()
print('Matrica AA: \n', AA)
print()

# Matrice kamera (u koordinatnom sistemu druge kamere)
CC = [EC[2, 1], EC[0, 2], EC[1, 0]]

CC1 = -AA.T @ CC

T1 = np.vstack(((K1 @ AA.T).T, K1 @ CC1)).T
T2 = np.hstack((K1, np.zeros((K1.shape[0], 1))))

print('Matrica prve kamere T1: \n', T1)
print()
print('Matrica druge kamere T2: \n', T2)
print()

def equations(T1, T2, m1, m2):
    return np.array([
         m1[1]*T1[2] - m1[2]*T1[1],
        -m1[0]*T1[2] + m1[2]*T1[0],
         m2[1]*T2[2] - m2[2]*T2[1],
        -m2[0]*T2[2] + m2[2]*T2[0]
    ])

def to_affine(mat):
    return mat[:-1] / mat[-1]

def triangulate(T1, T2, m1, m2):
    linear_system = equations(T1, T2, m1, m2)
    _, _, V = LA.svd(linear_system)
    M = to_affine(np.array(V[-1]))
    return M

points3D = [triangulate(T1, T2, m1, m2) for m1, m2 in zip(left_fixed, right_fixed)]

print('3D koordinate: ')
for point in points3D:
    print(point)


sides_cube_1 = [
    [points3D[0], points3D[1], points3D[2], points3D[3]],
    [points3D[4], points3D[5], points3D[6], points3D[7]],
    [points3D[0], points3D[1], points3D[5], points3D[4]],
    [points3D[3], points3D[2], points3D[6], points3D[7]],
    [points3D[1], points3D[2], points3D[6], points3D[5]],
    [points3D[0], points3D[3], points3D[7], points3D[4]]
]

sides_cube_2 = [
    [points3D[8], points3D[9], points3D[10], points3D[11]],
    [points3D[12], points3D[13], points3D[14], points3D[15]],
    [points3D[8], points3D[9], points3D[13], points3D[12]],
    [points3D[11], points3D[10], points3D[14], points3D[15]],
    [points3D[9], points3D[10], points3D[14], points3D[13]],
    [points3D[8], points3D[11], points3D[15], points3D[12]]
]

x_axis_camera = AA.T[0]
y_axis_camera = AA.T[1]
z_axis_camera = AA.T[2]

figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.add_collection3d(Poly3DCollection(sides_cube_1, facecolors='blue', linewidths=0.5, edgecolors='black', alpha=0.4))
axes.add_collection3d(Poly3DCollection(sides_cube_2, facecolors='red', linewidths=0.5, edgecolors='black', alpha=0.4))

axes.quiver(*CC, *x_axis_camera, color = 'blue')
axes.quiver(*CC, *y_axis_camera, color = 'green')
axes.quiver(*CC, *z_axis_camera, color = 'red')

axes.set_xlabel('X osa')
axes.set_ylabel('Y osa')
axes.set_zlabel('Z osa')

axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-2, 1])

plt.show()
