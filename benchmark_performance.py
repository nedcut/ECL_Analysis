#!/usr/bin/env python3
"""
Performance Benchmark Script for Brightness Sorcerer Optimizations

This script demonstrates the performance improvements made to the video brightness analysis tool.
It compares the original implementation with the optimized version.
"""

import time
import numpy as np
import cv2
import sys
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

# Add the current directory to path to import from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import BrightnessProcessor, OptimizedFrameCache, BATCH_SIZE

class PerformanceBenchmark:
    """Performance benchmarking for brightness analysis optimizations."""
    
    def __init__(self):
        self.test_results = {}
    
    def create_test_video_frames(self, num_frames: int = 1000, width: int = 1920, height: int = 1080) -> List[np.ndarray]:
        """Generate synthetic test frames for benchmarking."""
        print(f"Generating {num_frames} test frames ({width}x{height})...")
        frames = []
        
        for i in range(num_frames):
            # Create a frame with varying brightness patterns
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some brightness variation to make it realistic
            brightness_factor = 0.5 + 0.5 * np.sin(i * 0.01)  # Oscillating brightness
            frame = (frame * brightness_factor).astype(np.uint8)
            
            frames.append(frame)
        
        return frames
    
    def create_test_rois(self, width: int = 1920, height: int = 1080, num_rois: int = 8) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate test ROIs for benchmarking."""
        rois = []
        roi_width = width // 4
        roi_height = height // 4
        
        for i in range(num_rois):
            x = (i % 4) * roi_width
            y = (i // 4) * roi_height
            pt1 = (x, y)
            pt2 = (x + roi_width, y + roi_height)
            rois.append((pt1, pt2))
        
        return rois
    
    def benchmark_brightness_computation(self, frames: List[np.ndarray], rois: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> dict:
        """Benchmark brightness computation methods."""
        print("Benchmarking brightness computation...")
        
        # Extract ROIs from frames
        roi_samples = []
        for frame in frames[:100]:  # Use first 100 frames for ROI extraction
            for (pt1, pt2) in rois:
                x1, y1 = pt1
                x2, y2 = pt2
                roi = frame[y1:y2, x1:x2]
                roi_samples.append(roi)
        
        results = {}
        
        # Benchmark single ROI processing (original method)
        print("  Testing single ROI processing...")
        start_time = time.time()
        single_results = []
        for roi in roi_samples:
            result = BrightnessProcessor.compute_brightness_single(roi)
            single_results.append(result)
        single_time = time.time() - start_time
        results['single_processing'] = {
            'time': single_time,
            'rois_processed': len(roi_samples),
            'rois_per_second': len(roi_samples) / single_time
        }
        
        # Benchmark batch ROI processing (optimized method)
        print("  Testing batch ROI processing...")
        start_time = time.time()
        batch_results = BrightnessProcessor.compute_brightness_batch(roi_samples)
        batch_time = time.time() - start_time
        results['batch_processing'] = {
            'time': batch_time,
            'rois_processed': len(roi_samples),
            'rois_per_second': len(roi_samples) / batch_time
        }
        
        # Calculate speedup
        speedup = single_time / batch_time if batch_time > 0 else 1.0
        results['speedup'] = speedup
        
        print(f"  Single processing: {results['single_processing']['rois_per_second']:.1f} ROIs/sec")
        print(f"  Batch processing: {results['batch_processing']['rois_per_second']:.1f} ROIs/sec")
        print(f"  Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_frame_cache(self, frames: List[np.ndarray]) -> dict:
        """Benchmark frame caching performance."""
        print("Benchmarking frame cache performance...")
        
        cache = OptimizedFrameCache(max_size=200)
        results = {}
        
        # Test cache population
        print("  Testing cache population...")
        start_time = time.time()
        for i, frame in enumerate(frames[:200]):
            cache.put(i, frame)
        population_time = time.time() - start_time
        
        # Test cache retrieval (hits)
        print("  Testing cache hits...")
        start_time = time.time()
        hit_count = 0
        for i in range(200):
            frame = cache.get(i)
            if frame is not None:
                hit_count += 1
        hit_time = time.time() - start_time
        
        # Test cache misses
        print("  Testing cache misses...")
        start_time = time.time()
        miss_count = 0
        for i in range(200, 400):
            frame = cache.get(i)
            if frame is None:
                miss_count += 1
        miss_time = time.time() - start_time
        
        results = {
            'population_time': population_time,
            'population_fps': 200 / population_time,
            'hit_time': hit_time,
            'hit_rate': hit_count / 200,
            'miss_time': miss_time,
            'miss_rate': miss_count / 200,
            'cache_size': cache.get_size()
        }
        
        print(f"  Population: {results['population_fps']:.1f} fps")
        print(f"  Hit rate: {results['hit_rate']:.1%}")
        print(f"  Miss rate: {results['miss_rate']:.1%}")
        
        return results
    
    def benchmark_batch_processing(self, frames: List[np.ndarray], rois: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> dict:
        """Benchmark batch vs sequential frame processing."""
        print("Benchmarking batch processing...")
        
        test_frames = frames[:200]  # Use subset for testing
        results = {}
        
        # Sequential processing (original method)
        print("  Testing sequential processing...")
        start_time = time.time()
        sequential_results = []
        for frame in test_frames:
            frame_results = []
            fh, fw = frame.shape[:2]
            for (pt1, pt2) in rois:
                x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])
                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]
                    result = BrightnessProcessor.compute_brightness_single(roi)
                    frame_results.append(result)
            sequential_results.append(frame_results)
        sequential_time = time.time() - start_time
        
        # Batch processing (optimized method)
        print("  Testing batch processing...")
        start_time = time.time()
        batch_results = []
        
        for i in range(0, len(test_frames), BATCH_SIZE):
            batch_frames = test_frames[i:i + BATCH_SIZE]
            batch_frame_results = []
            
            for frame in batch_frames:
                frame_rois = []
                fh, fw = frame.shape[:2]
                for (pt1, pt2) in rois:
                    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
                    x2, y2 = min(fw - 1, pt2[0]), min(fh - 1, pt2[1])
                    if x2 > x1 and y2 > y1:
                        roi = frame[y1:y2, x1:x2]
                        frame_rois.append(roi)
                
                # Batch process all ROIs for this frame
                roi_results = BrightnessProcessor.compute_brightness_batch(frame_rois)
                batch_frame_results.append(roi_results)
            
            batch_results.extend(batch_frame_results)
        
        batch_time = time.time() - start_time
        
        # Calculate results
        frames_processed = len(test_frames)
        results = {
            'sequential_time': sequential_time,
            'sequential_fps': frames_processed / sequential_time,
            'batch_time': batch_time,
            'batch_fps': frames_processed / batch_time,
            'speedup': sequential_time / batch_time if batch_time > 0 else 1.0,
            'frames_processed': frames_processed
        }
        
        print(f"  Sequential: {results['sequential_fps']:.1f} fps")
        print(f"  Batch: {results['batch_fps']:.1f} fps")
        print(f"  Speedup: {results['speedup']:.2f}x")
        
        return results
    
    def run_full_benchmark(self, num_frames: int = 1000) -> dict:
        """Run complete performance benchmark."""
        print("=" * 60)
        print("BRIGHTNESS SORCERER PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Generate test data
        frames = self.create_test_video_frames(num_frames)
        rois = self.create_test_rois()
        
        # Run benchmarks
        brightness_results = self.benchmark_brightness_computation(frames, rois)
        cache_results = self.benchmark_frame_cache(frames)
        batch_results = self.benchmark_batch_processing(frames, rois)
        
        # Compile results
        all_results = {
            'brightness_computation': brightness_results,
            'frame_cache': cache_results,
            'batch_processing': batch_results,
            'test_parameters': {
                'num_frames': num_frames,
                'num_rois': len(rois),
                'frame_size': f"{frames[0].shape[1]}x{frames[0].shape[0]}"
            }
        }
        
        self.print_summary(all_results)
        return all_results
    
    def print_summary(self, results: dict):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        brightness = results['brightness_computation']
        cache = results['frame_cache']
        batch = results['batch_processing']
        params = results['test_parameters']
        
        print(f"Test Parameters:")
        print(f"  Frames: {params['num_frames']}")
        print(f"  ROIs: {params['num_rois']}")
        print(f"  Frame Size: {params['frame_size']}")
        print()
        
        print(f"Brightness Computation Optimization:")
        print(f"  Batch processing speedup: {brightness['speedup']:.2f}x")
        print(f"  ROI processing rate: {brightness['batch_processing']['rois_per_second']:.1f} ROIs/sec")
        print()
        
        print(f"Frame Cache Performance:")
        print(f"  Cache population rate: {cache['population_fps']:.1f} fps")
        print(f"  Cache hit rate: {cache['hit_rate']:.1%}")
        print(f"  Cache size: {cache['cache_size']} frames")
        print()
        
        print(f"Batch Processing Optimization:")
        print(f"  Frame processing speedup: {batch['speedup']:.2f}x")
        print(f"  Optimized processing rate: {batch['batch_fps']:.1f} fps")
        print()
        
        total_speedup = brightness['speedup'] * batch['speedup']
        print(f"ESTIMATED TOTAL SPEEDUP: {total_speedup:.2f}x")
        print("=" * 60)

def main():
    """Run the performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    # Allow command line argument for number of frames
    num_frames = 500  # Reduced for faster testing
    if len(sys.argv) > 1:
        try:
            num_frames = int(sys.argv[1])
        except ValueError:
            print("Invalid number of frames specified, using default 500")
    
    results = benchmark.run_full_benchmark(num_frames)
    
    # Optionally save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to 'benchmark_results.json'")

if __name__ == "__main__":
    main()