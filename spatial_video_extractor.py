import av
import numpy as np
import h5py
import json
import time
import argparse
from pathlib import Path

class DepthExtractor:
    """Extract depth data from iPhone spatial videos to HDF5 format"""
    
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.output_dir = self.video_path.parent / f"{self.video_path.stem}_depth_data"
        self.output_dir.mkdir(exist_ok=True)
        
    def extract(self):
        """Extract depth data and save to HDF5 file"""
        print(f"Processing video: {self.video_path}")
        
        container = av.open(str(self.video_path)) # open the video file
        
        print("\nAvailable streams:") # print available streams
        for i, stream in enumerate(container.streams):
            print(f"Stream #{i}: {stream.type} - Metadata: {stream.metadata}")
        
        depth_stream = None # find depth stream
        for stream in container.streams:
            if 'depth' in str(stream.metadata).lower():
                depth_stream = stream
                break
        
        if not depth_stream:
            raise ValueError("No depth stream found in video")
            
        depth_file_path = self.output_dir / f"{self.video_path.stem}_depth.h5" # create HDF5 file for depth data
        
        try:
            with h5py.File(depth_file_path, 'w') as f:
                depth_maps = f.create_group('depth_maps') # create datasets
                f.create_dataset('timestamps', (0,), maxshape=(None,), dtype=np.int64)
                
                frame_count = 0
                timestamps = []
                
                print("\nExtracting depth data...")
                for packet in container.demux(depth_stream):
                    for frame in packet.decode():
                        depth_map = np.frombuffer(frame.planes[0], dtype=np.float32) # convert depth frame to numpy array
                        depth_map = depth_map.reshape(frame.height, frame.width)
                        
                        depth_maps.create_dataset(
                            f'frame_{frame_count}',
                            data=depth_map,
                            compression='gzip',
                            compression_opts=9
                        )
                        
                        timestamps.append(frame.pts) # store timestamp
                        
                        frame_count += 1
                        if frame_count % 10 == 0:
                            print(f"Processed {frame_count} frames")
                
                f['timestamps'].resize((len(timestamps),)) # save timestamps
                f['timestamps'][:] = timestamps
                
                metadata = { # save metadata
                    'video_path': str(self.video_path),
                    'frame_count': frame_count,
                    'width': frame.width,
                    'height': frame.height,
                    'extraction_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'depth_stream_metadata': str(depth_stream.metadata)
                }
                f.attrs['metadata'] = json.dumps(metadata)
        
        finally:
            container.close()
            
        print(f"\nDepth data saved to: {depth_file_path}")
        return depth_file_path

def main():
    parser = argparse.ArgumentParser(description='Extract depth data from iPhone spatial video')
    parser.add_argument('video_path', help='Path to the spatial video file')
    args = parser.parse_args()
    
    extractor = DepthExtractor(args.video_path)
    depth_file_path = extractor.extract()
    
if __name__ == "__main__":
    main()