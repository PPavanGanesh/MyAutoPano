
import numpy as np
import cv2
import os
import imutils

def ANMS(corners, N_best):
    """
    Performs Adaptive Non-Maximal Suppression on corners from goodFeaturesToTrack
    """
    N_strong = len(corners)
    r = np.full(N_strong, np.inf)
    
    # Extract x,y coordinates
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]
    
    # For each corner, compute the minimum suppression radius
    for i in range(N_strong):
        for j in range(N_strong):
            if i != j:
                # Calculate Euclidean distance
                ED = (x_coords[j] - x_coords[i])**2 + (y_coords[j] - y_coords[i])**2
                if ED < r[i]:
                    r[i] = ED
    
    # Sort corners based on suppression radius
    sorted_indices = np.argsort(r)[::-1]
    selected_corners = corners[sorted_indices[:N_best]]
    
    return selected_corners

def detect_and_save_corners(images, output_path):
    """
    Detect and save corner visualizations for all images
    """
    os.makedirs(os.path.join(output_path, 'corners'), exist_ok=True)
    
    for idx, img in enumerate(images):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, 3000, 0.001, 8)
        
        # Visualize corners
        img_corners = img.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Save corner visualization
        cv2.imwrite(os.path.join(output_path, 'corners', f'corners_{idx}.jpg'), img_corners)

def detect_and_save_anms(images, output_path):
    """
    Apply ANMS and save visualizations for all images
    """
    os.makedirs(os.path.join(output_path, 'anms'), exist_ok=True)
    
    for idx, img in enumerate(images):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, 3000, 0.001, 8)
        
        # Apply ANMS
        selected_corners = ANMS(corners, 1500)
        
        # Visualize ANMS corners
        img_anms = img.copy()
        for corner in selected_corners:
            x, y = corner.ravel()
            cv2.circle(img_anms, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Save ANMS visualization
        cv2.imwrite(os.path.join(output_path, 'anms', f'anms_{idx}.jpg'), img_anms)

def create_and_visualize_feature_descriptors(img, gray, corners, output_path, filename, color=(0, 255, 0)):
    """
    Creates feature descriptors and visualizes the patches for each corner
    """
    os.makedirs(os.path.join(output_path, 'feature_descriptors'), exist_ok=True)
    
    descriptors = []
    valid_corners = []
    img_viz = img.copy()
    patch_size = 41
    half_size = patch_size // 2
    Rectangle_size = 5
    
    for idx, corner in enumerate(corners):
        x, y = corner.ravel()
        x, y = int(x), int(y)
        
        # Check boundaries
        if (x - half_size < 0 or x + half_size >= gray.shape[1] or 
            y - half_size < 0 or y + half_size >= gray.shape[0]):
            continue
            
        # Draw rectangle around the patch
        top_left = (x - Rectangle_size, y - Rectangle_size)
        bottom_right = (x + Rectangle_size, y + Rectangle_size)
        cv2.rectangle(img_viz, top_left, bottom_right, color, 1)
        
        # Extract and process patch
        patch = gray[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
        blurred = cv2.GaussianBlur(patch, (3,3), 0)
        subsampled = cv2.resize(blurred, (8,8))
        feature = subsampled.reshape(64,)
        
        # Standardize
        mean = np.mean(feature)
        std = np.std(feature)
        if std != 0:
            feature = (feature - mean) / std
            
        descriptors.append(feature)
        valid_corners.append(corner)
    
    # Save visualization
    output_filename = os.path.join(output_path, 'feature_descriptors', f'FD_{filename}')
    cv2.imwrite(output_filename, img_viz)
    
    return np.array(valid_corners), np.array(descriptors)

def stitch_middle_out(folder_path, output_path):
    """
    Stitch images starting from the middle and expanding outwards
    """
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
    
    n = len(images)
    if n < 2:
        print("Need at least 2 images for stitching")
        return
    
    # Find middle index
    mid = n // 2
    
    if n % 2 == 0:  # Even number of images
        # Start with middle two images
        result = stitch_pair(images[mid-1], images[mid], output_path, f'stitch_{mid-1}_{mid}')
        left_idx = mid - 2
        right_idx = mid + 1
    else:  # Odd number of images
        # Start with middle image
        result = images[mid]
        left_idx = mid - 1
        right_idx = mid + 1
    
    # Stitch outward alternately
    while left_idx >= 0 or right_idx < n:
        if left_idx >= 0:
            result = stitch_pair(images[left_idx], result, output_path, f'stitch_left_{left_idx}')
            left_idx -= 1
        
        if right_idx < n:
            result = stitch_pair(result, images[right_idx], output_path, f'stitch_right_{right_idx}')
            right_idx += 1
    
    cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), result)
    return result

############################################################################################################################1
# def stitch_middle_out(folder_path, output_path):
#     """
#     Stitch images starting from the middle and expanding outwards
#     First completes left side stitching, then right side
#     """
#     # Load all images
#     image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
#     images = []
#     for f in image_files:
#         img = cv2.imread(os.path.join(folder_path, f))
#         if img is not None:
#             images.append(img)
    
#     n = len(images)
#     if n < 2:
#         print("Need at least 2 images for stitching")
#         return
    
#     # Find middle index
#     mid = n // 2
    
#     if n % 2 == 0:  # Even number of images
#         # Start with middle two images
#         result = stitch_pair(images[mid-1], images[mid], output_path, f'stitch_{mid-1}_{mid}')
        
#         # Complete left side first
#         for i in range(mid-2, -1, -1):
#             result = stitch_pair(images[i], result, output_path, f'stitch_left_{i}')
        
#         # Then complete right side
#         for i in range(mid+1, n):
#             result = stitch_pair(result, images[i], output_path, f'stitch_right_{i}')
            
#     else:  # Odd number of images
#         # Start with middle image
#         result = images[mid]
        
#         # Alternate between left and right
#         left_idx = mid - 1
#         right_idx = mid + 1
        
#         while left_idx >= 0 or right_idx < n:
#             if left_idx >= 0:
#                 result = stitch_pair(images[left_idx], result, output_path, f'stitch_left_{left_idx}')
#                 left_idx -= 1
            
#             if right_idx < n:
#                 result = stitch_pair(result, images[right_idx], output_path, f'stitch_right_{right_idx}')
#                 right_idx += 1
    
#     cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), result)
#     return result
###############################################################################################################################

# def stitch_middle_out(folder_path, output_path):
#     """
#     Stitch images starting from the middle and expanding outwards
#     First completes left side stitching, then sequentially stitches right side
#     """
#     # Load all images
#     image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
#     images = []
#     for f in image_files:
#         img = cv2.imread(os.path.join(folder_path, f))
#         if img is not None:
#             images.append(img)
    
#     n = len(images)
#     if n < 2:
#         print("Need at least 2 images for stitching")
#         return
    
#     # Find middle index
#     mid = n // 2
    
#     if n % 2 == 0:  # Even number of images
#         # Start with middle two images (3 and 4 for 8 images)
#         result = stitch_pair(images[mid-1], images[mid], output_path, f'stitch_{mid-1}_{mid}')
        
#         # Complete left side first (2, 1, 0)
#         for i in range(mid-2, -1, -1):
#             result = stitch_pair(images[i], result, output_path, f'stitch_left_{i}')
#             # Save intermediate result after completing left side
#             if i == 0:
#                 cv2.imwrite(os.path.join(output_path, 'left_side_complete.jpg'), result)
        
#         # Then sequentially stitch with right side images (5, 6, 7)
#         for i in range(mid+1, n):
#             result = stitch_pair(result, images[i], output_path, f'stitch_complete_with_{i}')
#             # Save intermediate result after each right side stitch
#             cv2.imwrite(os.path.join(output_path, f'with_image_{i}_complete.jpg'), result)
            
#     else:  # Odd number of images
#         # Start with middle image
#         result = images[mid]
        
#         # Complete left side first
#         for i in range(mid-1, -1, -1):
#             result = stitch_pair(images[i], result, output_path, f'stitch_left_{i}')
        
#         # Then complete right side sequentially
#         for i in range(mid+1, n):
#             result = stitch_pair(result, images[i], output_path, f'stitch_complete_with_{i}')
    
#     cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), result)
#     return result


def match_features(desc1, desc2, corners1, corners2, ratio_threshold=0.8):
    """
    Match features between two images using ratio test
    """
    matches = []
    
    for i in range(len(desc1)):
        distances = []
        
        # Compute SSD with all descriptors in second image
        for j in range(len(desc2)):
            ssd = np.sum((desc1[i] - desc2[j]) ** 2)
            distances.append((j, ssd))
        
        # Sort distances
        distances.sort(key=lambda x: x[1])
        
        # Apply ratio test
        if len(distances) >= 2:
            if distances[0][1] < ratio_threshold * distances[1][1]:
                matches.append((i, distances[0][0]))
    
    return matches

def apply_RANSAC(matches, corners1, corners2, threshold=8.0, max_iterations=2000):
    """
    Apply RANSAC to find best homography and inliers
    """
    if len(matches) < 4:
        return None, []
        
    best_inliers = []
    best_H = None
    
    # Extract matched points
    points1 = np.float32([corners1[m[0]][0] for m in matches])
    points2 = np.float32([corners2[m[1]][0] for m in matches])
    
    for _ in range(max_iterations):
        # Randomly select 4 point pairs
        idx = np.random.choice(len(matches), 4, replace=False)
        sample_points1 = points1[idx]
        sample_points2 = points2[idx]
        
        try:
            H = cv2.getPerspectiveTransform(sample_points1, sample_points2)
            
            # Check all points
            inliers = []
            for i, (p1, p2) in enumerate(zip(points1, points2)):
                p1_transformed = cv2.perspectiveTransform(
                    np.array([[p1]]), H)[0][0]
                error = np.sqrt(np.sum((p2 - p1_transformed) ** 2))
                
                if error < threshold:
                    inliers.append(matches[i])
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H
                
        except:
            continue
    
    return best_H, best_inliers

# def stitch_images(img1, img2, H):
#     """
#     Stitch two images using homography and blend them
#     """
#     # Get dimensions
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
    
#     # Find corners of second image in first image space
#     corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
#     transformed_corners = cv2.perspectiveTransform(corners2, H)
#     corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
#     # Find dimensions of stitched image
#     all_corners = np.concatenate((corners1, transformed_corners), axis=0)
#     x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
#     x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
#     # Create translation matrix
#     translation = [-x_min, -y_min]
#     H_translation = np.array([[1, 0, translation[0]], 
#                             [0, 1, translation[1]], 
#                             [0, 0, 1]], dtype=np.float32)
    
#     # Warp images
#     output_size = (x_max - x_min, y_max - y_min)
#     warped_img2 = cv2.warpPerspective(img2, H_translation, output_size)
#     warped_img1 = cv2.warpPerspective(img1, H_translation.dot(H), output_size)
    
#     # Create masks for blending
#     mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
#     mask2 = (cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    
#     # Apply Gaussian blur to masks for smooth blending
#     mask1 = cv2.GaussianBlur(mask1, (5,5), 4)
#     mask2 = cv2.GaussianBlur(mask2, (5,5), 4)
    
#     # Normalize masks
#     sum_masks = mask1 + mask2
#     mask1 = mask1 / (sum_masks + 1e-6)
#     mask2 = mask2 / (sum_masks + 1e-4)
    
#     # Blend images
#     result = np.zeros_like(warped_img1, dtype=np.float32)
#     for c in range(3):
#         result[..., c] = (warped_img1[..., c] * mask1 + 
#                          warped_img2[..., c] * mask2)
    
#     return result.astype(np.uint8)

def stitch_images(img1, img2, H):
    """
    Memory-efficient version of image stitching
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((corners1, transformed_corners), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Check and limit output size
    max_dimension = 8000  # Maximum allowed dimension
    if x_max - x_min > max_dimension or y_max - y_min > max_dimension:
        scale = max_dimension / max(x_max - x_min, y_max - y_min)
        x_max = int(x_min + (x_max - x_min) * scale)
        y_max = int(y_min + (y_max - y_min) * scale)
        
        # Scale the images
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)
        H = H.copy()
        H[:2] *= scale
    
    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], 
                            [0, 1, translation[1]], 
                            [0, 0, 1]], dtype=np.float32)
    
    output_size = (x_max - x_min, y_max - y_min)
    
    # Process images in chunks
    chunk_size = 1000
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    for y in range(0, output_size[1], chunk_size):
        height = min(chunk_size, output_size[1] - y)
        chunk_img1 = cv2.warpPerspective(img1, H_translation.dot(H), 
                                       output_size, 
                                       dst=None,
                                       borderMode=cv2.BORDER_TRANSPARENT)
        chunk_img2 = cv2.warpPerspective(img2, H_translation, 
                                       output_size,
                                       dst=None,
                                       borderMode=cv2.BORDER_TRANSPARENT)
        
        # Create masks for current chunk
        mask1 = (cv2.cvtColor(chunk_img1[y:y+height], cv2.COLOR_BGR2GRAY) > 0)
        mask2 = (cv2.cvtColor(chunk_img2[y:y+height], cv2.COLOR_BGR2GRAY) > 0)
        
        # Blend the current chunk
        mask_overlap = mask1 & mask2
        result[y:y+height][mask1] = chunk_img1[y:y+height][mask1]
        result[y:y+height][mask2 & ~mask1] = chunk_img2[y:y+height][mask2 & ~mask1]
        
        # Blend overlapping regions
        if np.any(mask_overlap):
            blend = cv2.addWeighted(chunk_img1[y:y+height], 0.5, 
                                  chunk_img2[y:y+height], 0.5, 0)
            result[y:y+height][mask_overlap] = blend[mask_overlap]
        
        # Free memory
        del chunk_img1, chunk_img2, mask1, mask2
    
    return result


def stitch_pair(img1, img2, output_path, prefix):
    """
    Stitch a pair of images using the existing pipeline
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    corners1 = cv2.goodFeaturesToTrack(gray1, 3000, 0.001, 8)
    corners2 = cv2.goodFeaturesToTrack(gray2, 3000, 0.001, 8)

    # Save corner detection results
    img1_corners = img1.copy()
    img2_corners = img2.copy()
    for corner in corners1:
        x, y = corner.ravel()
        cv2.circle(img1_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
    for corner in corners2:
        x, y = corner.ravel()
        cv2.circle(img2_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
    
    cv2.imwrite(os.path.join(output_path, f'corners_{prefix}_1.jpg'), img1_corners)
    cv2.imwrite(os.path.join(output_path, f'corners_{prefix}_2.jpg'), img2_corners)

    # Apply ANMS
    selected_corners1 = ANMS(corners1, 1500)
    selected_corners2 = ANMS(corners2, 1500)

    img1_anms = img1.copy()
    img2_anms = img2.copy()
    for corner in selected_corners1:
        x, y = corner.ravel()
        cv2.circle(img1_anms, (int(x), int(y)), 2, (0, 255, 0), -1)
    for corner in selected_corners2:
        x, y = corner.ravel()
        cv2.circle(img2_anms, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    cv2.imwrite(os.path.join(output_path, f'anms_{prefix}_1.jpg'), img1_anms)
    cv2.imwrite(os.path.join(output_path, f'anms_{prefix}_2.jpg'), img2_anms)
    
    # Create feature descriptors
    valid_corners1, desc1 = create_and_visualize_feature_descriptors(
        img1, gray1, selected_corners1, output_path, f'{prefix}_desc1.jpg')
    valid_corners2, desc2 = create_and_visualize_feature_descriptors(
        img2, gray2, selected_corners2, output_path, f'{prefix}_desc2.jpg')
    
    # Match features
    matches = match_features(desc1, desc2, valid_corners1, valid_corners2)
    
    # Visualize feature matches
    visualize_matches(img1, img2, valid_corners1, valid_corners2, matches, output_path, f'matches_{prefix}.jpg')
    
    # Apply RANSAC
    H, inliers = apply_RANSAC(matches, valid_corners1, valid_corners2)
    
    # Visualize RANSAC inliers
    visualize_ransac_inliers(img1, img2, valid_corners1, valid_corners2, matches, 
                             [matches.index(m) for m in inliers], output_path, f'ransac_{prefix}.jpg')
    
    if H is not None:
        # Stitch images
        result = stitch_images(img1, img2, H)
        cv2.imwrite(os.path.join(output_path, f'{prefix}_result.jpg'), result)
        return result
    else:
        print(f"Failed to compute homography for {prefix}")
        return img1



def process_panorama(folder_path, output_path):
    """
    Main function to process panorama with comprehensive feature extraction
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    images = [cv2.imread(os.path.join(folder_path, f)) for f in image_files]
    
    # Detect and save corners for all images
    detect_and_save_corners(images, output_path)
    
    # Apply and save ANMS for all images
    detect_and_save_anms(images, output_path)
    
    # Process panorama using middle-out approach
    result = stitch_middle_out(folder_path, output_path)
    
    if result is not None:
        print("Panorama creation completed successfully")
    else:
        print("Failed to create panorama")

def visualize_matches(img1, img2, corners1, corners2, matches, output_path, filename):
    """
    Visualize feature matches between two images
    """
    os.makedirs(os.path.join(output_path, 'matches'), exist_ok=True)
    
    # Create a side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    # Draw matches
    for m in matches:
        # Get matched corner coordinates
        (x1, y1) = map(int, corners1[m[0]].ravel())
        (x2, y2) = map(int, corners2[m[1]].ravel())
        
        # Adjust x2 coordinate to account for first image's width
        x2 += w1

        # Draw lines connecting matched points
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)

    # Save visualization
    output_filename = os.path.join(output_path, 'matches', filename)
    cv2.imwrite(output_filename, vis)

def visualize_ransac_inliers(img1, img2, corners1, corners2, matches, inliers, output_path, filename):
    """
    Visualize RANSAC inliers and outliers
    """
    os.makedirs(os.path.join(output_path, 'ransac'), exist_ok=True)
    
    # Create a side-by-side image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2

    # Convert matches to sets for efficient lookup
    inlier_set = set(inliers)
    
    # Draw all matches first (as potential matches)
    for i, m in enumerate(matches):
        (x1, y1) = map(int, corners1[m[0]].ravel())
        (x2, y2) = map(int, corners2[m[1]].ravel())
        x2 += w1

        # Differentiate inliers and outliers
        if i in inlier_set:
            # Inliers in green
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (x1, y1), 4, (0, 255, 0), -1)
            cv2.circle(vis, (x2, y2), 4, (0, 255, 0), -1)
        # else:
            # Outliers in red
            # cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
            # cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)

    # Save visualization
    output_filename = os.path.join(output_path, 'ransac', filename)
    cv2.imwrite(output_filename, vis)
    
def main():
    # # Set your input and output paths
    # folder_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Phase1\Data\Train\Set1"
    # output_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Code\output\OP_1"

    folder_path = r"C:\Users\pavan\OneDrive\Desktop\Phase1\Code\Data\Test\TestSet4"
    output_path = r"C:\Users\pavan\OneDrive\Desktop\Phase1\Code\Outputs\Test_Set4"
      
    
    # Process panorama
    process_panorama(folder_path, output_path)
    # process_panorama_1(folder_path,output_path)

if __name__ == "__main__":
    main()
