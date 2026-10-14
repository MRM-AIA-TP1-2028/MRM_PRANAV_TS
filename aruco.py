import cv2
import numpy as np

# ==============================================================================
# CONFIGURATION & CAMERA CALIBRATION PARAMETERS
# ==============================================================================
# Real-world physical side length of your printed ArUco marker in meters or mm.
# Note: Position (tvec) and Depth units match this measurement unit.
MARKER_LENGTH = 0.05  # e.g., 0.05 meters (5 cm)

# Multi-dictionary mapping for 4x4, 5x5, and 6x6 tag support
ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}

def get_camera_matrix(frame_width, frame_height):
    """
    Generates an estimated camera matrix (K) and zero distortion coefficients.
    Replace these with actual intrinsic values from camera calibration for higher precision.
    """
    focal_length = frame_width  # Approximate focal length in pixels
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def detect_and_estimate_pose():
    # Initialize Laptop Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

    # Read a sample frame to get video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture initial frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    camera_matrix, dist_coeffs = get_camera_matrix(w, h)

    # Instantiate detector parameters
    detector_params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters') else cv2.aruco.DetectorParameters_create()

    print("--- ArUco Pose & Depth Estimation Running ---")
    print("Supports: 4x4, 5x5, and 6x6 ArUco Dictionaries.")
    print("Press 'q' or 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Iterate through requested dictionaries (4x4, 5x5, 6x6)
        for dict_name, dict_id in ARUCO_DICTIONARIES.items():
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id) if hasattr(cv2.aruco, 'getPredefinedDictionary') else cv2.aruco.Dictionary_get(dict_id)

            # Detect markers
            if hasattr(cv2.aruco, 'ArucoDetector'):
                # OpenCV 4.7+ API
                detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
                corners, ids, rejected = detector.detectMarkers(gray)
            else:
                # Legacy OpenCV API
                corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

            if ids is not None and len(ids) > 0:
                # Draw green bounding boxes and ID overlays
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Estimate pose for each detected marker
                for i in range(len(ids)):
                    marker_corners = corners[i]

                    # 3D object points for a square marker centered at origin
                    obj_points = np.array([
                        [-MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
                        [ MARKER_LENGTH / 2,  MARKER_LENGTH / 2, 0],
                        [ MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
                        [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]
                    ], dtype=np.float32)

                    # Estimate pose using Perspective-n-Point (PnP)
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        marker_corners[0],
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:
                        # Draw 3D coordinate axes on top of marker (X=Red, Y=Green, Z=Blue)
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH * 0.6)

                        # Calculate Euclidean Depth and Position Coordinates
                        tx, ty, tz = tvec.flatten()
                        depth = np.sqrt(tx**2 + ty**2 + tz**2)

                        # Determine top-left corner position for visual telemetry display
                        c_corner = marker_corners[0][0].astype(int)
                        text_x, text_y = c_corner[0], c_corner[1] - 10

                        # Overlay Pose Data & Depth Stream onto Video Frame
                        info_str = f"ID: {ids[i][0]} ({dict_name[5:8]}) | Depth: {depth:.2f}m"
                        pos_str = f"X:{tx:.2f} Y:{ty:.2f} Z:{tz:.2f}"

                        cv2.putText(frame, info_str, (text_x, max(text_y, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, pos_str, (text_x, max(text_y + 18, 38)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Render output display window
        cv2.imshow("ArUco Pose & Depth Estimation", frame)

        # Break loop on 'q' or 'Esc' keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_estimate_pose()