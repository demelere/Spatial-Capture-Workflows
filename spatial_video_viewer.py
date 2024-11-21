import rerun as rr
import numpy as np
import cv2
from pathlib import Path
from enum import Enum
import colorsys

class DepthVisualizationMode(Enum):
    RAINBOW = "rainbow"
    HEATMAP = "heatmap"
    GRAYSCALE = "grayscale"
    POINT_CLOUD = "point_cloud"

class SpatialVideoVisualizer:
    def __init__(self, video_path, vis_mode=DepthVisualizationMode.RAINBOW):
        """
        Initialize the spatial video visualizer
        
        Args:
            video_path (str): Path to the spatial video file
            vis_mode (DepthVisualizationMode): Visualization mode for depth data
        """
        self.video_path = video_path
        self.vis_mode = vis_mode
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Initialize rerun with recording capability
        rr.init("Spatial Video Viewer", recording_id="spatial_recording")
        rr.spawn()
        
        # Set up the visualization space
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    
    def create_rainbow_depth(self, depth_data):
        """Convert depth data to rainbow colormap"""
        normalized = cv2.normalize(depth_data, None, 0, 1, cv2.NORM_MINMAX)
        colored = np.zeros((*depth_data.shape, 3), dtype=np.uint8)
        
        for i in range(depth_data.shape[0]):
            for j in range(depth_data.shape[1]):
                if normalized[i, j] > 0:  # Skip invalid depth values
                    rgb = colorsys.hsv_to_rgb(normalized[i, j], 1.0, 1.0)
                    colored[i, j] = np.array(rgb) * 255
                    
        return colored
    
    def create_point_cloud(self, depth_data, rgb_frame):
        """Create point cloud data from depth and RGB"""
        height, width = depth_data.shape
        fx = width  # Approximate focal length, adjust based on actual camera params
        fy = height
        cx = width / 2
        cy = height / 2
        
        points = []
        colors = []
        
        for v in range(height):
            for u in range(width):
                depth = depth_data[v, u]
                if depth > 0:  # Valid depth
                    # Convert from image coordinates to 3D coordinates
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    
                    points.append([x, y, z])
                    colors.append(rgb_frame[v, u] / 255.0)  # Normalize color values
        
        return np.array(points), np.array(colors)
    
    def visualize_frame(self, frame_rgb, depth_frame, frame_num):
        """
        Visualize a single frame with its depth data
        
        Args:
            frame_rgb (np.ndarray): RGB frame
            depth_frame (np.ndarray): Depth frame
            frame_num (int): Frame number
        """
        # Log the original RGB frame
        rr.log("world/video", rr.Image(frame_rgb))
        
        # Process and visualize depth data based on selected mode
        if self.vis_mode == DepthVisualizationMode.RAINBOW:
            depth_vis = self.create_rainbow_depth(depth_frame)
            rr.log("world/depth/rainbow", rr.Image(depth_vis))
            
        elif self.vis_mode == DepthVisualizationMode.HEATMAP:
            depth_vis = cv2.applyColorMap(
                cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            rr.log("world/depth/heatmap", rr.Image(depth_vis))
            
        elif self.vis_mode == DepthVisualizationMode.GRAYSCALE:
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rr.log("world/depth/grayscale", rr.Image(depth_vis))
            
        elif self.vis_mode == DepthVisualizationMode.POINT_CLOUD:
            points, colors = self.create_point_cloud(depth_frame, frame_rgb)
            rr.log("world/depth/point_cloud", 
                   rr.Points3D(points, colors=colors, radii=0.01))
            
        # Log depth metrics
        depth_stats = {
            "min": np.min(depth_frame[depth_frame > 0]),
            "max": np.max(depth_frame),
            "mean": np.mean(depth_frame[depth_frame > 0]),
            "std": np.std(depth_frame[depth_frame > 0])
        }
        
        for stat_name, value in depth_stats.items():
            rr.log(f"world/depth/stats/{stat_name}", 
                   rr.Scalar(value))
    
    def process_video(self):
        """Process and visualize the entire video"""
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Get depth data (placeholder - implement based on your format)
                depth_frame = self.get_depth_data(frame_count)
                self.visualize_frame(frame_rgb, depth_frame, frame_count)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        self.cap.release()
    
    def get_depth_data(self, frame_count):
        """
        Placeholder for depth data extraction.
        Implement based on your spatial video format.
        """
        # For testing, generate synthetic depth data
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Create a radial gradient for testing
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        depth = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        depth = cv2.normalize(depth, None, 0, 10, cv2.NORM_MINMAX)
        
        return depth

def main():
    # Replace with your spatial video path
    VIDEO_PATH = "IMG_6540.MOV"
    
    # Create visualizer with desired visualization mode
    visualizer = SpatialVideoVisualizer(
        VIDEO_PATH,
        vis_mode=DepthVisualizationMode.RAINBOW  # Try different modes
    )
    
    # Process the video
    visualizer.process_video()

if __name__ == "__main__":
    main()