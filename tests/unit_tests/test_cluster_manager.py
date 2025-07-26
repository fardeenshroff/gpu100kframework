"""
Unit tests for ClusterManager
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cluster_manager import ClusterManager, TrainingJob

def test_cluster_manager_init():
    """Test ClusterManager initialization"""
    manager = ClusterManager("configs/cluster_configs/nvidia_100k.yaml")
    assert manager.cluster_health == "HEALTHY"
    assert isinstance(manager.active_jobs, dict)

def test_training_job_creation():
    """Test TrainingJob dataclass"""
    job = TrainingJob("test_id", "resnet50", "imagenet", 8, True)
    assert job.id == "test_id"
    assert job.model == "resnet50"
    assert job.gpus == 8
    assert job.fault_tolerance == True

def test_job_submission():
    """Test job submission"""
    manager = ClusterManager("configs/cluster_configs/nvidia_100k.yaml")
    job = manager.submit_job("test_model", "test_dataset", 16)
    assert job.model == "test_model"
    assert job.gpus == 16
    assert job.id in manager.active_jobs
