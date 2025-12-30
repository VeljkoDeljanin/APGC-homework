import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def center(T):
    C1 = LA.det(np.delete(T, 0, 1))
    C2 = LA.det(np.delete(T, 1, 1))
    C3 = LA.det(np.delete(T, 2, 1))
    C4 = LA.det(np.delete(T, 3, 1))

    C = np.array([C1, -C2, C3, -C4]) / -C4

    C = np.where(np.isclose(C, 0), 0.0, C)

    return C

def camera_K(T):
    T0 = np.delete(T, 3, 1)
    if LA.det(T0) <= 0:
        T *= -1
    
    T0_inv = LA.inv(T0)

    Q, R = LA.qr(T0_inv)

    if R[0][0] < 0:
        R = np.matmul(np.diag([-1, 1, 1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))
    if R[1][1] < 0:
        R = np.matmul(np.diag([1, -1, 1]), R)
        Q = np.matmul(Q, np.diag([1, -1, 1]))
    if R[2][2] < 0:
        R = np.matmul(np.diag([1, 1, -1]), R)
        Q = np.matmul(Q, np.diag([1, 1, -1]))

    K = LA.inv(R)
    K = K / K[-1, -1]
    K = np.where(np.isclose(K, 0), 0.0, K)

    return K

def camera_A(T):
    T0 = np.delete(T, 3, 1)
    if LA.det(T0) <= 0:
        T *= -1
    
    T0_inv = LA.inv(T0)

    Q, R = LA.qr(T0_inv)

    if R[0][0] <= 0:
        R = np.matmul(np.diag([-1, 1, 1]), R)
        Q = np.matmul(Q, np.diag([-1, 1, 1]))
    if R[1][1] <= 0:
        R = np.matmul(np.diag([1, -1, 1]), R)
        Q = np.matmul(Q, np.diag([1, -1, 1]))
    if R[2][2] <= 0:
        R = np.matmul(np.diag([1, 1, -1]), R)
        Q = np.matmul(Q, np.diag([1, 1, -1]))

    A = Q
    A = np.where(np.isclose(A, 0), 0.0, A)

    return A

def two_equations(img, orig):
    O4 = np.zeros(4)
    row1 = np.concatenate((O4, -img[2]*orig, img[1]*orig))
    row2 = np.concatenate((img[2]*orig, O4, -img[0]*orig))
    return np.array([row1, row2])

def camera_matrix(imgs, origs):
    A = []

    for img, orig in zip(imgs, origs):
        rows = two_equations(img, orig)
        A.append(rows[0])
        A.append(rows[1])

    A = np.array(A)

    _, _, Vh = LA.svd(A)
    T = Vh[-1].reshape(3, 4)

    T = T / T[-1, -1]
    T = np.where(np.isclose(T, 0), 0.0, T)

    return T

if __name__ == '__main__':
    # 1600 x 1200
    imgs = np.array([1, -1, -1])*np.array([1600, 0, 0]) - np.array([
                                                                    [803, 372, 1],
                                                                    [795, 624, 1],
                                                                    [981, 483, 1],
                                                                    [1014, 240, 1],
                                                                    [775, 138, 1],
                                                                    [560, 247, 1],
                                                                    [596, 492, 1],
                                                                    [789, 283, 1]
                                                                ])

    origs = np.array([
        [3, 3, 3, 1],
        [3, 3, 0, 1],
        [0, 3, 0, 1],
        [0, 3, 3, 1],
        [0, 0, 3, 1],
        [3, 0, 3, 1],
        [3, 0, 0, 1],
        [2, 2, 3, 1]
    ])

    vertices = np.array([[3, 0, 0], [3, 3, 0], [0, 3, 0], [0, 0, 0], [3, 0, 3], [3, 3, 3], [0, 3, 3], [0, 0, 3]])

    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]], # up
        [vertices[0], vertices[1], vertices[5], vertices[4]], # front left
        [vertices[1], vertices[2], vertices[6], vertices[5]], # front right
        [vertices[0], vertices[3], vertices[7], vertices[4]], # back left
        [vertices[2], vertices[3], vertices[7], vertices[6]]  # back right
    ]

    origin = np.array([0, 0, 0])

    x_box_axis = np.array([10, 0, 0])
    y_box_axis = np.array([0, 10, 0])
    z_box_axis = np.array([0, 0, 10])

    T = camera_matrix(imgs, origs)

    C = center(T)
    K = camera_K(T)
    A = camera_A(T)

    print('Matrica kamere:\n', T)
    print()
    print('Matrica kalibracije kamere:\n', K)
    print()
    print('Pozicija centra kamere:\n', C)
    print()
    print('Spoljasnja matrica kamere:\n', A)

    x_camera_axis = A[0]
    y_camera_axis = A[1]
    z_camera_axis = A[2]
    
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.add_collection3d(Poly3DCollection(edges, facecolors='orchid', linewidths=0.5, edgecolors='black', alpha=1))

    axes.quiver(*origin, *x_box_axis, color='blue')
    axes.quiver(*origin, *y_box_axis, color='green')
    axes.quiver(*origin, *z_box_axis, color='red')

    axes.quiver(*C[:3], *x_camera_axis, color='blue')
    axes.quiver(*C[:3], *y_camera_axis, color='green')
    axes.quiver(*C[:3], *z_camera_axis, color='red')

    axes.set_xlabel('X axis')
    axes.set_ylabel('Y axis')
    axes.set_zlabel('Z axis')

    axes.set_xlim([-1, 15])
    axes.set_ylim([-1, 15])
    axes.set_zlim([0, 15])

    plt.show()
