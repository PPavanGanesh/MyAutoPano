
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
        corners = cv2.goodFeaturesToTrack(gray, 2500, 0.001, 8)
        
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
        corners = cv2.goodFeaturesToTrack(gray, 1600, 0.001, 8)
        
        # Apply ANMS
        selected_corners = ANMS(corners, 600)
        
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

# def stitch_middle_out(folder_path, output_path):
#     """
#     Stitch images starting from the middle and expanding outwards
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
#         left_idx = mid - 2
#         right_idx = mid + 1
#     else:  # Odd number of images
#         # Start with middle image
#         result = images[mid]
#         left_idx = mid - 1
#         right_idx = mid + 1
    
#     # Stitch outward alternately
#     while left_idx >= 0 or right_idx < n:
#         if left_idx >= 0:
#             result = stitch_pair(images[left_idx], result, output_path, f'stitch_left_{left_idx}')
#             left_idx -= 1
        
#         if right_idx < n:
#             result = stitch_pair(result, images[right_idx], output_path, f'stitch_right_{right_idx}')
#             right_idx += 1
    
#     cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), result)
#     return result
def stitch_multiple_images2(images, output_path, min_matches=10):
    """
    Stitch multiple images using pairwise RANSAC with global refinement
    """
    n = len(images)
    if n < 2:
        return None
            
    # First pass: compute pairwise homographies
    homographies = []
    for i in range(n-1):
        # Convert to grayscale
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i+1], cv2.COLOR_BGR2GRAY)
        
        # Detect corners with increased parameters
        c1 = cv2.goodFeaturesToTrack(gray1, 3500, 0.001, 8)
        c2 = cv2.goodFeaturesToTrack(gray2, 3500, 0.001, 8)
        
        # Apply ANMS
        c1 = ANMS(c1, 2500)
        c2 = ANMS(c2, 2500)
        
        # Get descriptors
        valid_corners1, desc1 = create_and_visualize_feature_descriptors(
            images[i], gray1, c1, output_path, f'desc_{i}.jpg')
        valid_corners2, desc2 = create_and_visualize_feature_descriptors(
            images[i+1], gray2, c2, output_path, f'desc_{i+1}.jpg')
        
        matches = match_features(desc1, desc2, valid_corners1, valid_corners2)
        if len(matches) < min_matches:
            print(f"Not enough matches between images {i} and {i+1}")
            continue
            
        H, inliers = apply_RANSAC(matches, valid_corners1, valid_corners2)
        if H is not None:
            homographies.append((i, i+1, H))
    
    # Stitch using computed homographies
    result = images[0]
    for i, j, H in homographies:
        result = stitch_images(result, images[j], H)
        
    return result


def stitch_middle_out2(folder_path, output_path):
    """
    Stitch images starting from the middle and expanding outwards with robust error handling
    """
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    print(f"Found {len(image_files)} images to process")
    
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
            print(f"Loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")
    
    n = len(images)
    if n < 2:
        print("Need at least 2 images for stitching")
        return None
    
    print(f"\nStarting stitching process with {n} images")
    mid = n // 2
    
    # Try different starting points if middle stitching fails
    if n % 2 == 0:
        print(f"\nAttempting to stitch middle pair: {image_files[mid-1]} and {image_files[mid]}")
        result = stitch_pair(images[mid-1], images[mid], output_path, f'stitch_{mid-1}_{mid}')
        
        if result is None:
            # Try alternative starting points
            for offset in range(1, mid):
                print(f"Middle stitch failed, trying pair: {image_files[mid-offset-1]} and {image_files[mid-offset]}")
                result = stitch_pair(images[mid-offset-1], images[mid-offset], output_path, f'stitch_{mid-offset-1}_{mid-offset}')
                if result is not None:
                    mid = mid - offset
                    break
        
        if result is None:
            print("Failed to find a suitable starting pair")
            return None
            
        left_idx = mid - 2
        right_idx = mid + 1
    else:
        print(f"\nStarting with middle image: {image_files[mid]}")
        result = images[mid]
        left_idx = mid - 1
        right_idx = mid + 1
    
    # Stitch outward with progress tracking
    print("\nProceeding with outward stitching:")
    total_remaining = (left_idx + 1) + (n - right_idx)
    stitches_completed = 0
    
    while left_idx >= 0 or right_idx < n:
        if left_idx >= 0:
            print(f"Stitching left image {image_files[left_idx]} ({stitches_completed + 1}/{total_remaining})")
            temp_result = stitch_pair(images[left_idx], result, output_path, f'stitch_left_{left_idx}')
            if temp_result is not None:
                result = temp_result
                stitches_completed += 1
            else:
                print(f"Warning: Failed to stitch {image_files[left_idx]}, skipping")
            left_idx -= 1
        
        if right_idx < n:
            print(f"Stitching right image {image_files[right_idx]} ({stitches_completed + 1}/{total_remaining})")
            temp_result = stitch_pair(result, images[right_idx], output_path, f'stitch_right_{right_idx}')
            if temp_result is not None:
                result = temp_result
                stitches_completed += 1
            else:
                print(f"Warning: Failed to stitch {image_files[right_idx]}, skipping")
            right_idx += 1
    
    if stitches_completed > 0:
        print(f"\nStitching complete! Successfully stitched {stitches_completed}/{total_remaining} images")
        print(f"Saving final panorama to: {os.path.join(output_path, 'final_panorama.jpg')}")
        cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), result)
        return result
    else:
        print("\nFailed to create panorama: No successful stitches")
        return None

def stitch_sequential(folder_path, output_path):
    """
    Stitch images sequentially from left to right with adaptive parameters
    """
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    print(f"Found {len(image_files)} images to process")
    
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
            print(f"Loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")
    
    n = len(images)
    if n < 2:
        print("Need at least 2 images for stitching")
        return None
    
    print(f"\nStarting sequential stitching process with {n} images")
    
    # Initialize with first image
    result = images[0]
    stitches_completed = 0
    
    # Stitch remaining images sequentially
    for i in range(1, n):
        print(f"\nStitching image pair {i}/{n-1}: {image_files[i-1]} -> {image_files[i]}")
        
        # Increase parameters progressively
        base_corners = 6000 + (i * 1000)  # Start at 3500, increase by 500 each time
        base_anms = 4500 + (i * 1000)     # Start at 2500, increase by 500 each time
        
        print(f"Using {base_corners} corners and {base_anms} ANMS points")
        
        temp_result = stitch_pair(
            result, 
            images[i], 
            output_path, 
            f'stitch_{i}',
            base_corners=base_corners,
            base_anms=base_anms,
            increment=500
        )
        
        if temp_result is not None:
            result = temp_result
            stitches_completed += 1
            print(f"Successfully stitched pair {i}")
            # Save intermediate panorama
            cv2.imwrite(os.path.join(output_path, f'panorama_progress_{i}.jpg'), result)
        else:
            print(f"Warning: Failed to stitch image {i+1}, attempting to continue with next image")
            
        # Force memory cleanup
        cv2.waitKey(1)
    
    if stitches_completed > 0:
        print(f"\nStitching complete! Successfully stitched {stitches_completed}/{n-1} pairs")
        final_path = os.path.join(output_path, 'final_panorama.jpg')
        print(f"Saving final panorama to: {final_path}")
        cv2.imwrite(final_path, result)
        return result
    else:
        print("\nFailed to create panorama: No successful stitches")
        return None



def stitch_middle_out1(folder_path, output_path):
    """
    Stitch images starting from the middle and expanding outwards
    """
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    print(f"Found {len(image_files)} images to process")
    
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
            print(f"Loaded image: {f}")
    
    n = len(images)
    if n < 2:
        print("Need at least 2 images for stitching")
        return
    
    print(f"\nStarting stitching process with {n} images")
    # Find middle index
    mid = n // 2
    
    if n % 2 == 0:  # Even number of images
        # Start with middle two images
        print(f"\nStarting with middle pair of images:")
        print(f"Stitching {image_files[mid-1]} and {image_files[mid]}")
        result = stitch_pair(images[mid-1], images[mid], output_path, f'stitch_{mid-1}_{mid}')
        left_idx = mid - 2
        right_idx = mid + 1
    else:  # Odd number of images
        # Start with middle image
        print(f"\nStarting with middle image: {image_files[mid]}")
        result = images[mid]
        left_idx = mid - 1
        right_idx = mid + 1
    
    # Stitch outward alternately
    print("\nProceeding with outward stitching:")
    while left_idx >= 0 or right_idx < n:
        if left_idx >= 0:
            print(f"Stitching left image {image_files[left_idx]} to current panorama")
            result = stitch_pair(images[left_idx], result, output_path, f'stitch_left_{left_idx}')
            left_idx -= 1
        
        if right_idx < n:
            print(f"Stitching right image {image_files[right_idx]} to current panorama")
            result = stitch_pair(result, images[right_idx], output_path, f'stitch_right_{right_idx}')
            right_idx += 1
    
    print("\nStitching complete!")
    print(f"Saving final panorama to: {os.path.join(output_path, 'final_panorama.jpg')}")
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
def apply_RANSAC(matches, corners1, corners2, threshold=12.0, max_iterations=4000, min_inliers=10):
    """
    Apply RANSAC to find best homography and inliers
    Args:
        matches: List of matching point pairs
        corners1, corners2: Corner points from both images
        threshold: Distance threshold for inlier classification
        max_iterations: Maximum RANSAC iterations
        min_inliers: Minimum number of inliers required
    Returns:
        best_H: Best homography matrix found, or None if no good match
        best_inliers: List of inlier matches
    """
    # Check if we have enough matches
    if len(matches) < 4:
        print("Not enough matches to compute homography")
        return None, []
    
    best_inliers = []
    best_H = None
    
    # Extract matched points
    try:
        points1 = np.float32([corners1[m[0]][0] for m in matches])
        points2 = np.float32([corners2[m[1]][0] for m in matches])
    except IndexError:
        print("Invalid corner indices in matches")
        return None, []
    
    # Ensure points are properly shaped for perspective transform
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        points1 = points1.reshape(-1, 1, 2)
        points2 = points2.reshape(-1, 1, 2)
    
    for _ in range(max_iterations):
        # Randomly select 4 point pairs
        if len(matches) < 4:
            break
            
        idx = np.random.choice(len(matches), 4, replace=False)
        sample_points1 = points1[idx]
        sample_points2 = points2[idx]
        
        try:
            # Ensure points are in correct format for getPerspectiveTransform
            H = cv2.getPerspectiveTransform(
                sample_points1.reshape(4, 2).astype(np.float32),
                sample_points2.reshape(4, 2).astype(np.float32)
            )
            
            # Check all points
            inliers = []
            for i, (p1, p2) in enumerate(zip(points1, points2)):
                # Reshape point for transformation
                p1_reshaped = p1.reshape(-1, 1, 2)
                try:
                    p1_transformed = cv2.perspectiveTransform(
                        p1_reshaped, H)[0][0]
                    error = np.sqrt(np.sum((p2.ravel() - p1_transformed) ** 2))
                    
                    if error < threshold:
                        inliers.append(matches[i])
                except:
                    continue
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H
                
        except cv2.error:
            continue
        except np.linalg.LinAlgError:
            continue
    
    # Check if we found a good enough solution
    if len(best_inliers) < min_inliers:
        print(f"Not enough inliers found (minimum {min_inliers} required)")
        return None, []
        
    return best_H, best_inliers


# def apply_RANSAC(matches, corners1, corners2, threshold=3.0, max_iterations=2000):
#     """
#     Apply RANSAC to find best homography and inliers
#     """
#     if len(matches) < 4:
#         return None, []
        
#     best_inliers = []
#     best_H = None
    
#     # Extract matched points
#     points1 = np.float32([corners1[m[0]][0] for m in matches])
#     points2 = np.float32([corners2[m[1]][0] for m in matches])
    
#     for _ in range(max_iterations):
#         # Randomly select 4 point pairs
#         idx = np.random.choice(len(matches), 4, replace=False)
#         sample_points1 = points1[idx]
#         sample_points2 = points2[idx]
        
#         try:
#             H = cv2.getPerspectiveTransform(sample_points1, sample_points2)
            
#             # Check all points
#             inliers = []
#             for i, (p1, p2) in enumerate(zip(points1, points2)):
#                 p1_transformed = cv2.perspectiveTransform(
#                     np.array([[p1]]), H)[0][0]
#                 error = np.sqrt(np.sum((p2 - p1_transformed) ** 2))
                
#                 if error < threshold:
#                     inliers.append(matches[i])
            
#             if len(inliers) > len(best_inliers):
#                 best_inliers = inliers
#                 best_H = H
                
#         except:
#             continue
    
#     return best_H, best_inliers

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
    Enhanced image stitching with better black region removal and memory management
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Find corners of warped image
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Calculate output dimensions
    all_corners = np.concatenate((corners1, transformed_corners), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Handle large panoramas
    max_dimension = 12000  # Increased max dimension
    if x_max - x_min > max_dimension or y_max - y_min > max_dimension:
        scale = max_dimension / max(x_max - x_min, y_max - y_min)
        x_max = int(x_min + (x_max - x_min) * scale)
        y_max = int(y_min + (y_max - y_min) * scale)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale)
        H = H.copy()
        H[:2] *= scale
    
    # Translation matrix
    translation = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation[0]], 
                            [0, 1, translation[1]], 
                            [0, 0, 1]], dtype=np.float32)
    
    output_size = (x_max - x_min, y_max - y_min)
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    # Process in smaller chunks to manage memory
    chunk_size = 1500  # Increased chunk size
    for y in range(0, output_size[1], chunk_size):
        height = min(chunk_size, output_size[1] - y)
        
        # Warp images
        chunk_img1 = cv2.warpPerspective(img1, H_translation.dot(H), 
                                       output_size, 
                                       dst=None,
                                       borderMode=cv2.BORDER_TRANSPARENT)
        chunk_img2 = cv2.warpPerspective(img2, H_translation, 
                                       output_size,
                                       dst=None,
                                       borderMode=cv2.BORDER_TRANSPARENT)
        
        # Create masks
        mask1 = (cv2.cvtColor(chunk_img1[y:y+height], cv2.COLOR_BGR2GRAY) > 0)
        mask2 = (cv2.cvtColor(chunk_img2[y:y+height], cv2.COLOR_BGR2GRAY) > 0)
        
        # Blend images
        mask_overlap = mask1 & mask2
        result[y:y+height][mask1] = chunk_img1[y:y+height][mask1]
        result[y:y+height][mask2 & ~mask1] = chunk_img2[y:y+height][mask2 & ~mask1]
        
        if np.any(mask_overlap):
            # Enhanced blending for overlapping regions
            blend = cv2.addWeighted(chunk_img1[y:y+height], 0.5, 
                                  chunk_img2[y:y+height], 0.5, 0)
            result[y:y+height][mask_overlap] = blend[mask_overlap]
        
        # Clean up memory
        del chunk_img1, chunk_img2, mask1, mask2
    
    # Remove black regions
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Add padding to avoid cutting edges
        padding = 20  # Increased padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(result.shape[1] - x, w + 2*padding)
        h = min(result.shape[0] - y, h + 2*padding)
        
        # Crop to content
        result = result[y:y+h, x:x+w]
        
        # Additional cleanup
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask = result_gray > 0
        if not np.all(mask):
            # Fill small black holes
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            result[~mask.astype(bool)] = 0
    
    return result


def stitch_images3(img1, img2, H):
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


# def stitch_pair(img1, img2, output_path, prefix):
#     """
#     Stitch a pair of images using the existing pipeline
#     """
#     # Convert to grayscale
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     # Detect corners
#     corners1 = cv2.goodFeaturesToTrack(gray1, 3500, 0.001, 8)
#     corners2 = cv2.goodFeaturesToTrack(gray2, 3500, 0.001, 8)

#     # Save corner detection results
#     img1_corners = img1.copy()
#     img2_corners = img2.copy()
#     for corner in corners1:
#         x, y = corner.ravel()
#         cv2.circle(img1_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
#     for corner in corners2:
#         x, y = corner.ravel()
#         cv2.circle(img2_corners, (int(x), int(y)), 2, (0, 0, 255), -1)
    
#     cv2.imwrite(os.path.join(output_path, f'corners_{prefix}_1.jpg'), img1_corners)
#     cv2.imwrite(os.path.join(output_path, f'corners_{prefix}_2.jpg'), img2_corners)

#     # Apply ANMS
#     selected_corners1 = ANMS(corners1, 2500)
#     selected_corners2 = ANMS(corners2, 2500)

#     img1_anms = img1.copy()
#     img2_anms = img2.copy()
#     for corner in selected_corners1:
#         x, y = corner.ravel()
#         cv2.circle(img1_anms, (int(x), int(y)), 2, (0, 255, 0), -1)
#     for corner in selected_corners2:
#         x, y = corner.ravel()
#         cv2.circle(img2_anms, (int(x), int(y)), 2, (0, 255, 0), -1)
    
#     cv2.imwrite(os.path.join(output_path, f'anms_{prefix}_1.jpg'), img1_anms)
#     cv2.imwrite(os.path.join(output_path, f'anms_{prefix}_2.jpg'), img2_anms)
    
#     # Create feature descriptors
#     valid_corners1, desc1 = create_and_visualize_feature_descriptors(
#         img1, gray1, selected_corners1, output_path, f'{prefix}_desc1.jpg')
#     valid_corners2, desc2 = create_and_visualize_feature_descriptors(
#         img2, gray2, selected_corners2, output_path, f'{prefix}_desc2.jpg')
    
#     # Match features
#     matches = match_features(desc1, desc2, valid_corners1, valid_corners2)
    
#     # Visualize feature matches
#     visualize_matches(img1, img2, valid_corners1, valid_corners2, matches, output_path, f'matches_{prefix}.jpg')
    
#     # Apply RANSAC
#     H, inliers = apply_RANSAC(matches, valid_corners1, valid_corners2)
    
#     # Visualize RANSAC inliers
#     visualize_ransac_inliers(img1, img2, valid_corners1, valid_corners2, matches, 
#                              [matches.index(m) for m in inliers], output_path, f'ransac_{prefix}.jpg')
    
#     if H is not None:
#         # Stitch images
#         result = stitch_images(img1, img2, H)
#         cv2.imwrite(os.path.join(output_path, f'{prefix}_result.jpg'), result)
#         return result
#     else:
#         print(f"Failed to compute homography for {prefix}")
#         return img1

def stitch_pair(img1, img2, output_path, prefix, base_corners=4500, base_anms=3500, increment=750):
    """
    Stitch a pair of images with incrementally increasing feature points
    """
    # Extract stitch number from prefix for incremental adjustment
    stitch_number = int(prefix.split('_')[-1]) if prefix.split('_')[-1].isdigit() else 0
    current_corners = base_corners + (stitch_number * increment)
    current_anms = base_anms + (stitch_number * increment)
    
    print(f"Using {current_corners} corners and {current_anms} ANMS points for {prefix}")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Detect corners with increased parameters
    corners1 = cv2.goodFeaturesToTrack(gray1, current_corners, 0.001, 8)
    corners2 = cv2.goodFeaturesToTrack(gray2, current_corners, 0.001, 8)
    
    if corners1 is None or corners2 is None:
        print(f"Failed to detect corners for {prefix}")
        return img1

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

    # Apply ANMS with increased points
    selected_corners1 = ANMS(corners1, current_anms)
    selected_corners2 = ANMS(corners2, current_anms)

    # Visualize ANMS results
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
    
    if len(valid_corners1) < 4 or len(valid_corners2) < 4:
        print(f"Not enough valid corners for {prefix}")
        return img1
    
    # Match features with adaptive ratio threshold
    ratio_threshold = 0.8 - (stitch_number * 0.05)  # Decrease threshold for later stitches
    ratio_threshold = max(0.6, ratio_threshold)  # Don't go below 0.6
    matches = match_features(desc1, desc2, valid_corners1, valid_corners2, ratio_threshold)
    
    if len(matches) < 10:
        print(f"Not enough matches found for {prefix}")
        return img1
    
    # Visualize feature matches
    visualize_matches(img1, img2, valid_corners1, valid_corners2, matches, 
                     output_path, f'matches_{prefix}.jpg')
    
    # Apply RANSAC with adaptive threshold
    ransac_threshold = 12.0 + (stitch_number * 2)  # Increase threshold for later stitches
    H, inliers = apply_RANSAC(matches, valid_corners1, valid_corners2, 
                             threshold=ransac_threshold, 
                             max_iterations=3000,
                             min_inliers=10 + stitch_number)
    
    if inliers:
        # Visualize RANSAC inliers
        visualize_ransac_inliers(img1, img2, valid_corners1, valid_corners2, matches, 
                                [matches.index(m) for m in inliers], 
                                output_path, f'ransac_{prefix}.jpg')
    
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
    Main function to process panorama
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process panorama using sequential approach
    # result = stitch_sequential(folder_path, output_path)
    result = stitch_middle_sections(folder_path, output_path)
    if result is not None:
        print("Panorama creation completed successfully")
    else:
        print("Failed to create panorama")
        
def process_panorama2(folder_path, output_path):
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

def stitch_middle_sections(folder_path, output_path):
    """
    Stitch images in three sections: middle-left, middle-right, and middle,
    then combine them all
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
    
    n = len(images)
    if n < 3:
        print("Need at least 3 images for sectional stitching")
        return None
    
    mid = n // 2
    print(f"\nProcessing {n} images in sections")
    
    # Process middle-left section
    print("\nProcessing middle-left section...")
    left_result = images[mid]
    for i in range(mid-1, -1, -1):
        print(f"Stitching image {i} to middle-left section")
        temp_result = stitch_pair(images[i], left_result, output_path, f'left_section_{i}')
        if temp_result is not None:
            left_result = temp_result
            cv2.imwrite(os.path.join(output_path, 'middle_left_section.jpg'), left_result)
    
    # Process middle-right section
    print("\nProcessing middle-right section...")
    right_result = images[mid]
    for i in range(mid+1, n):
        print(f"Stitching image {i} to middle-right section")
        temp_result = stitch_pair(right_result, images[i], output_path, f'right_section_{i}')
        if temp_result is not None:
            right_result = temp_result
            cv2.imwrite(os.path.join(output_path, 'middle_right_section.jpg'), right_result)
    
    # Process middle section (using middle three images)
    print("\nProcessing middle section...")
    if mid > 0 and mid < n-1:
        middle_result = stitch_pair(images[mid-1], images[mid], output_path, 'middle_section_1')
        if middle_result is not None:
            middle_result = stitch_pair(middle_result, images[mid+1], output_path, 'middle_section_2')
            if middle_result is not None:
                cv2.imwrite(os.path.join(output_path, 'middle_section.jpg'), middle_result)
    
    # Final combination
    print("\nCombining all sections...")
    # Combine left and middle sections
    if left_result is not None and middle_result is not None:
        final_result = stitch_pair(left_result, middle_result, output_path, 'final_left_middle')
        # Combine with right section
        if final_result is not None and right_result is not None:
            final_result = stitch_pair(final_result, right_result, output_path, 'final_complete')
            if final_result is not None:
                cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), final_result)
                print("Successfully created final panorama")
                return final_result
    
    print("Failed to create complete panorama")
    return None
def stitch_middle_sections(folder_path, output_path):
    """
    Stitch images in three sections: middle-left, middle-right, and middle,
    then combine them all
    """
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    
    # Load all images
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    images = []
    for f in image_files:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is not None:
            images.append(img)
    
    n = len(images)
    if n < 3:
        print("Need at least 3 images for sectional stitching")
        return None
    
    mid = n // 2
    print(f"\nProcessing {n} images in sections")
    
    # Process middle-left section
    print("\nProcessing middle-left section...")
    left_result = images[mid]
    for i in range(mid-1, -1, -1):
        print(f"Stitching image {i} to middle-left section")
        temp_result = stitch_pair(images[i], left_result, output_path, f'left_section_{i}')
        if temp_result is not None:
            left_result = temp_result
            cv2.imwrite(os.path.join(output_path, 'middle_left_section.jpg'), left_result)
    
    # Process middle-right section
    print("\nProcessing middle-right section...")
    right_result = images[mid]
    for i in range(mid+1, n):
        print(f"Stitching image {i} to middle-right section")
        temp_result = stitch_pair(right_result, images[i], output_path, f'right_section_{i}')
        if temp_result is not None:
            right_result = temp_result
            cv2.imwrite(os.path.join(output_path, 'middle_right_section.jpg'), right_result)
    
    # Process middle section (using middle three images)
    print("\nProcessing middle section...")
    if mid > 0 and mid < n-1:
        middle_result = stitch_pair(images[mid-1], images[mid], output_path, 'middle_section_1')
        if middle_result is not None:
            middle_result = stitch_pair(middle_result, images[mid+1], output_path, 'middle_section_2')
            if middle_result is not None:
                cv2.imwrite(os.path.join(output_path, 'middle_section.jpg'), middle_result)
    
    # Final combination
    print("\nCombining all sections...")
    # Increase parameters for final combination
    final_corners = 6000  # Increased from 4500
    final_anms = 5000    # Increased from 3500
    
    # Combine left and middle sections with increased parameters
    if left_result is not None and middle_result is not None:
        # Use higher parameters for final stitching
        final_result = stitch_pair(
            left_result, 
            middle_result, 
            output_path, 
            'final_left_middle',
            base_corners=final_corners,
            base_anms=final_anms,
            increment=1000  # Larger increment for final stitching
        )
        
        # Combine with right section using even higher parameters
        if final_result is not None and right_result is not None:
            final_result = stitch_pair(
                final_result, 
                right_result, 
                output_path, 
                'final_complete',
                base_corners=final_corners + 1000,  # Further increase
                base_anms=final_anms + 1000,       # Further increase
                increment=1500                      # Even larger increment
            )
            
            if final_result is not None:
                cv2.imwrite(os.path.join(output_path, 'final_panorama.jpg'), final_result)
                print("Successfully created final panorama")
                return final_result
    
    print("Failed to create complete panorama")
    return None

    
def main():
    # # Set your input and output paths
    folder_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Phase1\Data\Train\Set3"
    output_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Code\output\OP_New3.111"

    # folder_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Phase1\Data\Test\TestSet2"
    # output_path = r"C:\Users\duggi\OneDrive\Desktop\WPI\Subjects\Computer Vision\CV_P1\Pavan_code\Code\Final_Output\OUTPUT_TestSet33"
      
    
    # Process panorama
    process_panorama(folder_path, output_path)
    # process_panorama_1(folder_path,output_path)

if __name__ == "__main__":
    main()
