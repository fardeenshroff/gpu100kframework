"""
Unit tests for monitoring system
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from monitoring.cluster_monitor import ClusterMonitor, GPUMetrics

def test_cluster_monitor_init():
    """Test ClusterMonitor initialization"""
    config = {"cluster_size": 1000}
    monitor = ClusterMonitor(config)
    assert monitor.cluster_size == 1000
    assert monitor.monitoring_active == False

def test_gpu_metrics():
    """Test GPUMetrics dataclass"""
    metrics = GPUMetrics("gpu_1", 85.5, 70.2, 75.0, 250.0, "HEALTHY")
    assert metrics.gpu_id == "gpu_1"
    assert metrics.utilization == 85.5
    assert metrics.health_status == "HEALTHY"
