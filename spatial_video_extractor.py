import av
import numpy as np
import rerun as rr
from pathlib import Path
import json

class iPhoneSpatialVideoExtractor:
    """Extract RGB and depth data from iPhone spatial videos"""
    
    def __init__(self, video_path):
        self.container = av.open(video_path)
        
        # Get video streams
        self.streams = {
            stream.type: stream
            for stream in self.container.streams
        }
        
        # Print available streams for debugging
        print("Available streams:")
        for stream_type, stream in self.streams.items():
            print(f"- {stream_type}: {stream.metadata}")
    
    def extract_frames(self):
        """
        Extract synchronized RGB and depth frames from the spatial video.
        Returns RGB frames and depth maps that rerun can visualize.
        """
        # Get the main video stream
        video_stream = self.streams.get('video')
        if not video_stream:
            raise ValueError("No video stream found")
            
        # Get the depth stream (could be named differently depending on iPhone format)
        depth_stream = None
        for stream in self.container.streams:
            if 'depth' in str(stream.metadata).lower():
                depth_stream = stream
                break
        
        if not depth_stream:
            print("Warning: No depth stream found. Check if this is a spatial video.")
            
        # Prepare rerun logging
        rr.init("iPhone Spatial Video Viewer")
        rr.connect()
        
        frame_count = 0
        
        # Dictionary to store depth maps keyed by timestamp
        depth_maps = {}
        
        # First pass: collect depth data
        if depth_stream:
            for packet in self.container.demux(depth_stream):
                for frame in packet.decode():
                    # Convert depth frame to numpy array
                    depth_map = np.frombuffer(frame.planes[0], dtype=np.float32)
                    depth_map = depth_map.reshape(frame.height, frame.width)
                    
                    # Store with timestamp
                    depth_maps[frame.pts] = depth_map
        
        # Reset container
        self.container.seek(0)
        
        # Second pass: process RGB frames and match with depth
        for frame in self.container.decode(video=0):
            # Convert RGB frame
            rgb_frame = frame.to_ndarray(format='rgb24')
            
            # Log RGB frame to rerun
            rr.log("world/video", rr.Image(rgb_frame))
            
            # Find matching depth map
            depth_map = depth_maps.get(frame.pts)
            if depth_map is not None:
                # Log depth visualization options
                
                # 1. As depth image
                rr.log("world/depth/image", 
                      rr.DepthImage(depth_map, meter=True))
                
                # 2. As point cloud
                # Create point cloud from depth map
                h, w = depth_map.shape
                y, x = np.mgrid[0:h, 0:w]
                
                # Basic pinhole camera model (adjust parameters as needed)
                fx = w  # approximate focal length
                fy = h
                cx = w / 2
                cy = h / 2
                
                # Calculate 3D points
                x_3d = (x - cx) * depth_map / fx
                y_3d = (y - cy) * depth_map / fy
                z_3d = depth_map
                
                # Stack points and reshape
                points = np.stack([x_3d, y_3d, z_3d], axis=-1)
                valid_mask = depth_map > 0
                points = points[valid_mask]
                
                # Get colors for point cloud
                colors = rgb_frame[valid_mask] / 255.0
                
                # Log point cloud
                rr.log("world/depth/point_cloud",
                      rr.Points3D(points, colors=colors, radii=0.01))
                
                # 3. Log depth statistics
                valid_depths = depth_map[depth_map > 0]
                if len(valid_depths) > 0:
                    stats = {
                        "min": np.min(valid_depths),
                        "max": np.max(valid_depths),
                        "mean": np.mean(valid_depths),
                        "std": np.std(valid_depths)
                    }
                    
                    for stat_name, value in stats.items():
                        rr.log(f"world/depth/stats/{stat_name}",
                              rr.Scalar(value))
            
            frame_count += 1
            
            # Optional: progress indicator
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")

def main():
    # Replace with your spatial video path
    VIDEO_PATH = "IMG_6540.MOV"
    
    # Create extractor
    extractor = iPhoneSpatialVideoExtractor(VIDEO_PATH)
    
    # Extract and visualize frames
    extractor.extract_frames()

if __name__ == "__main__":
    main()