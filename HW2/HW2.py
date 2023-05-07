import cv2
import numpy as np
import random
import math
import sys
import os

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, grays, SIFT_Detector, threshold = 0.75):
        # Step1 - extract the keypoints and features by SIFT
        key_points_1, descriptors_1 = SIFT_Detector.detectAndCompute(grays[0], None)
        key_points_2, descriptors_2 = SIFT_Detector.detectAndCompute(grays[1], None)
        
        # Step2 - extract the match point with threshold (David Lowe's ratio test)
        matches = self.matchKeyPoint(key_points_1, descriptors_1, key_points_2, descriptors_2, threshold)
        
        # Step3 - fit the homography model with RANSAC algorithm
        H = self.RANSAC_get_H(matches)

        # Step4 - Warp image to create panoramic image
        warp_img = self.warp(imgs[0], imgs[1], H)
        
        return warp_img

    def matchKeyPoint(self, kps_1, features_1, kps_2, features_2, threshold):
        '''
        Match the Keypoints beteewn two image
        '''
        matches = []
        for i in range(len(features_1)):
            min_index, min_distance = -1, np.inf
            sec_index, sec_distance = -1, np.inf
            
            for j in range(len(features_2)):
                distance = np.linalg.norm(features_1[i] - features_2[j])
                
                if distance < min_distance:
                    sec_index, sec_distance = min_index, min_distance
                    min_index, min_distance = j, distance
                    
                elif distance < sec_distance and sec_index != min_index:
                    sec_index, sec_distance = j, distance
                    
            matches.append([min_index, min_distance, sec_index, sec_distance])

        good_matches = []
        for i in range(len(matches)):
            if matches[i][1] <= matches[i][3] * threshold:
                good_matches.append([(int(kps_1[i].pt[0]), int(kps_1[i].pt[1])), 
                                     (int(kps_2[matches[i][0]].pt[0]), int(kps_2[matches[i][0]].pt[1]))])
        
        return good_matches
    
    def RANSAC_get_H(self, matches):
        img1_kp = []
        img2_kp = []
        for kp1, kp2 in matches:
            img1_kp.append(list(kp1))
            img2_kp.append(list(kp2))
        img1_kp = np.array(img1_kp)
        img2_kp = np.array(img2_kp)
        
        homography = Homography()
        threshold = 5
        iteration_num = 8000
        max_inliner_num = 0
        best_H = None
        
        for iter in range(iteration_num):
            random_sample_idx = random.sample(range(len(matches)), 4)
            H = homography.solve_homography(img1_kp[random_sample_idx], img2_kp[random_sample_idx])

            # find the best Homography have the the maximum number of inlier
            inliner_num = 0
            
            for i in range(len(matches)):
                if i not in random_sample_idx:
                    concateCoor = np.hstack((img1_kp[i], [1])) # add z-axis as 1
                    dstCoor = H @ concateCoor.T
                    
                    # avoid divide zero number, or too small number cause overflow
                    if dstCoor[2] <= 1e-8: 
                        continue
                    
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - img2_kp[i]) < threshold):
                        inliner_num = inliner_num + 1
            
            if (max_inliner_num < inliner_num):
                max_inliner_num = inliner_num
                best_H = H

        return best_H
                
    def warp(self, img1, img2, H):
        left_down = np.hstack(([0], [0], [1]))
        left_up = np.hstack(([0], [img1.shape[0]-1], [1]))
        right_down = np.hstack(([img1.shape[1]-1], [0], [1]))
        right_up = np.hstack(([img1.shape[1]-1], [img1.shape[0]-1], [1]))
        
        warped_left_down = H @ left_down.T
        warped_left_up = H @ left_up.T
        warped_right_down =  H @ right_down.T
        warped_right_up = H @ right_up.T

        x1 = int(min(min(min(warped_left_down[0],warped_left_up[0]),min(warped_right_down[0], warped_right_up[0])), 0))
        y1 = int(min(min(min(warped_left_down[1],warped_left_up[1]),min(warped_right_down[1], warped_right_up[1])), 0))
        size = (img2.shape[1] + abs(x1), img2.shape[0] + abs(y1))

        A = np.float32([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]])
        warped1 = cv2.warpPerspective(src=img1, M=A@H, dsize=size)
        warped2 = cv2.warpPerspective(src=img2, M=A, dsize=size)
        
        blender = Blender()
        result = blender.linearBlendingWithConstantWidth([warped1, warped2])

        return result

class Blender:
    def __init__(self):
        pass
    
    def linearBlending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(img_left_mask[i, j]) > 0):
                    linearBlending_img[i, j] = img_left[i, j]
                else:
                    linearBlending_img[i, j] = img_right[i, j]
        return linearBlending_img
    
    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 3 # constant width
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
                    
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr):
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            
            # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
            middleIdx = int((maxIdx + minIdx) / 2)
            
            # left 
            for j in range(minIdx, middleIdx + 1):
                if (j >= middleIdx - constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 1
            # right
            for j in range(middleIdx + 1, maxIdx + 1):
                if (j <= middleIdx + constant_width):
                    alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                else:
                    alpha_mask[i, j] = 0

        
        linearBlendingWithConstantWidth_img = np.copy(img_right)
        linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
        # linear blending with constant width
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
                elif(np.count_nonzero(img_left_mask[i, j]) > 0):
                    linearBlendingWithConstantWidth_img[i, j] = img_left[i, j]
                else:
                    linearBlendingWithConstantWidth_img[i, j] = img_right[i, j]
        return linearBlendingWithConstantWidth_img

class Homography:
    def __init__(self):
        pass
    
    def solve_homography(self, kps_1, kps_2):
        A = []
        for i in range(len(kps_1)):
            A.append([kps_1[i, 0], kps_1[i, 1], 1, 0, 0, 0, -kps_1[i, 0] * kps_2[i, 0], -kps_1[i, 1] * kps_2[i, 0], -kps_2[i, 0]])
            A.append([0, 0, 0, kps_1[i, 0], kps_1[i, 1], 1, -kps_1[i, 0] * kps_2[i, 1], -kps_1[i, 1] * kps_2[i, 1], -kps_2[i, 1]])

        # Solve system of linear equations Ah = 0 using SVD
        u, sigma, vt = np.linalg.svd(A)
        
        # pick H from last line of vt
        H = np.reshape(vt[8], (3, 3))
        
        # normalization, let H[2,2] equals to 1
        H = (1/H.item(8)) * H
        
        return H

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()
    
    SIFT_Detector = cv2.SIFT_create()
    stitcher = Stitcher()
    fileList = ['baseline', 'bonus']
    
    for filename in fileList:
        if filename == 'baseline':
            imgNameList = ['m1.jpg', 'm2.jpg', 'm3.jpg', 'm4.jpg', 'm5.jpg', 'm6.jpg']
        else:
            imgNameList = ['m1.jpg', 'm2.jpg', 'm3.jpg', 'm4.jpg']

        img1, img_gray1 = read_img(os.path.join(filename, imgNameList[0]))
        img2, img_gray2 = read_img(os.path.join(filename, imgNameList[1]))
        
        result_img = stitcher.stitch([img1, img2], [img_gray1, img_gray2],SIFT_Detector, threshold = 0.75)
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        
        
        for img_name in imgNameList[2:]:
            next_img, next_img_gray = read_img(os.path.join(filename, img_name))
            result_img = stitcher.stitch([result_img, next_img], [result_gray, next_img_gray], SIFT_Detector, threshold = 0.75)
            result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
            
        cv2.imwrite(os.path.join(filename, 'result2.jpg'), result_img)
        

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)