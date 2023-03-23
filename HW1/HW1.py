import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import normalize

image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z, filepath):
    Z_map = np.reshape(Z, (image_row, image_col)).copy()
    data = np.zeros((image_row * image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image


def read_data(filepath):
    L = []
    I = []
    num = 0
    
    # Get the light source from LightSource.txt
    with open(filepath + '/LightSource.txt', 'r') as f: 
        for line in f.readlines():
            line = line.strip()
            point = list(map(int, line[line.find('(') + 1: line.find(')')].split(',')))
            L.append(np.array(point).astype(np.float32))
            num += 1
    
    # Get the "unit vector" of light source, dim of L: (m, 3), m is number of light source
    L = normalize(L, axis = 1)
    
    # Get image data
    for i in range(1, num + 1):
        pic = read_bmp(filepath + '/pic' + str(i) + '.bmp')
        # Flatten image size from H * W to HW
        I.append(pic.ravel()) 

    # Change type to ndarray
    I = np.asarray(I)

    return I, L

def get_normal_vector(I, L):
    # I = L * Kd * N, solve the equation to get Kd * N
    KdN = np.linalg.solve(L.T @ L, L.T @ I)
    KdN = normalize(KdN, axis = 0)

    return KdN.T


def fill_value_into_matrix(M, v, s, N, mask):
    nonzero_h, nonzero_w = np.where(mask!=0)
    
    # Calculate the index number used in filling M
    index_array = np.zeros((image_row, image_col)).astype(np.int16)
    for i in range(s):
        index_array[nonzero_h[i], nonzero_w[i]] = i
    
    for i in range(s):
        h = nonzero_h[i]
        w = nonzero_w[i]
        n_x = N[h, w, 0]
        n_y = N[h, w, 1]
        n_z = N[h, w, 2]
        
        # z(x+1, y) - z(x, y) = - nx / nz
        j = i*2
        if mask[h, w+1]:
            k = index_array[h, w+1]
            M[j, i] = -1
            M[j, k] = 1
            v[j] = -n_x/n_z
        elif mask[h, w-1]:
            k = index_array[h, w-1]
            M[j, k] = -1
            M[j, i] = 1
            v[j] = -n_x/n_z
        
        # z(x, y+1) - z(x, y) = -ny / nz
        j = i*2+1
        if mask[h+1, w]:   
            k = index_array[h+1, w]
            M[j, i] = 1
            M[j, k] = -1
            v[j] = -n_y/n_z
        elif mask[h-1, w]:
            k = index_array[h, w-1]
            M[j, k] = 1
            M[j, i] = -1
            v[j] = -n_y/n_z
            
    return M, v      

def get_surface_reconstruction(z, mask, s, optimize):
    nonzero_h, nonzero_w = np.where(mask!=0)
    
    # Filter strange point in z
    normalized_z = (z - np.mean(z)) / np.std(z)
    outliner_idx = np.abs(normalized_z) > 2
    z_max = np.max(z[~outliner_idx])
    z_min = np.min(z[~outliner_idx])
    
    Z = mask.astype(np.float32)
    
    if optimize:
        for i in range(s):
            if z[i] > z_max:
                Z[nonzero_h[i], nonzero_w[i]] = z_max
            elif z[i] < z_min:
                Z[nonzero_h[i], nonzero_w[i]] = z_min
            else:
                Z[nonzero_h[i], nonzero_w[i]] = z[i]
    else:
        for i in range(s):
            Z[nonzero_h[i], nonzero_w[i]] = z[i]
        
    return Z

def compute_depth(mask, N, optimize = True):
    N = np.reshape(N, (image_row, image_col, 3))
    
    # number of pixels of the object 
    s = np.size(np.where(mask != 0)[0])
    
    # Mz = V
    M = scipy.sparse.lil_matrix((2*s, s))
    v = np.zeros((2*s, 1))
    
    # Fill the value into M and v
    M, v = fill_value_into_matrix(M, v, s, N, mask)

    # M.T * M * z = M.T * v
    z = scipy.sparse.linalg.spsolve(M.T @ M, M.T @ v)
    
    return get_surface_reconstruction(z, mask, s, optimize)
    
    

if __name__ == '__main__':
    objects = ['star', 'bunny', 'venus']
    
    for object in objects:
        filepath = 'test/' + object
    
        I, L = read_data(filepath)
        N = get_normal_vector(I, L)
        normal_visualization(N)
        
        mask = read_bmp(filepath + '/pic1.bmp')
        Z = compute_depth(mask, N, optimize = True)
        
        depth_visualization(Z)
        save_ply(Z, filepath + '/' + object + '.ply')
        show_ply(filepath + '/' + object + '.ply')

        # showing the windows of all visualization function
        plt.show()