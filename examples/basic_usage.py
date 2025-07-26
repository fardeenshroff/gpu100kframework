#!/usr/bin/env python3
"""
Basic Usage Example - gpu100kframework
Demonstrates core functionality for NVIDIA-scale operations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cluster_manager import ClusterManager
import time

def main():
    print("🚀 gpu100kframework - Basic Usage Demo")
    print("Target: NVIDIA 100K+ GPU AI Supercomputers\n")
    
    # Initialize cluster manager
    print("Step 1: Initializing Cluster Manager...")
    cluster = ClusterManager(config_file="configs/cluster_configs/nvidia_100k.yaml")
    
    # Start monitoring
    print("\nStep 2: Starting Monitoring System...")
    cluster.start_monitoring()
    
    # Enable fault tolerance
    print("\nStep 3: Enabling Fault Tolerance...")
    cluster.enable_fault_tolerance()
    
    # Submit a training job
    print("\nStep 4: Submitting AI Training Job...")
    job = cluster.submit_job(
        model="resnet50",
        dataset="imagenet",
        gpus=1024,
        fault_tolerance=True
    )
    
    # Monitor the job
    print("\nStep 5: Monitoring Job Progress...")
    for i in range(3):
        time.sleep(2)
        cluster.monitor_job(job.id)
        print()
    
    print("🎯 Demo completed successfully!")
    print("✅ Framework ready for NVIDIA AI Resiliency Engineering")

if __name__ == "__main__":
    main()
