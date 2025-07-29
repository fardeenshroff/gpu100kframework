# src/gpu_cluster_optimizer/core/types.py
"""
Core type definitions for GPU Cluster Optimizer
Enterprise-grade type safety and validation
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


class OptimizationStrategy(str, Enum):
    """Supported optimization strategies"""
    BASELINE = "baseline"
    NEOCPU_INSPIRED = "neocpu_inspired"
    TVM_INSPIRED = "tvm_inspired"
    HYBRID_OPTIMIZATION = "hybrid_optimization"


class GPUStatus(str, Enum):
    """GPU operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPUSpecification(BaseModel):
    """GPU hardware specification with validation"""
    
    model: str = Field(..., description="GPU model name")
    memory_gb: int = Field(..., gt=0, le=256, description="GPU memory in GB")
    compute_capability: float = Field(..., gt=0.0, description="Compute capability in TFLOPS")
    tensor_cores: bool = Field(default=False, description="Has tensor core support")
    nvlink_support: bool = Field(default=False, description="Supports NVLink")
    
    @validator('memory_gb')
    def validate_memory(cls, v):
        if v not in [8, 16, 24, 32, 40, 48, 80, 128]:
            raise ValueError("Memory must be standard GPU memory size")
        return v


class GPUNode(BaseModel):
    """Production-grade GPU node representation"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Human-readable node name")
    specification: GPUSpecification
    status: GPUStatus = Field(default=GPUStatus.HEALTHY)
    current_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_celsius: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    power_consumption_watts: Optional[float] = Field(default=None, ge=0.0)
    
    # Network topology information
    rack_id: Optional[str] = Field(default=None)
    pod_id: Optional[str] = Field(default=None)
    cluster_id: str = Field(..., description="Cluster identifier")
    
    # Monitoring metadata
    last_health_check: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class WorkloadRequirements(BaseModel):
    """Workload resource requirements specification"""
    
    memory_gb: int = Field(..., gt=0, description="Required memory in GB")
    compute_tflops: float = Field(..., gt=0.0, description="Required compute in TFLOPS")
    min_gpus: int = Field(default=1, ge=1, description="Minimum GPUs required")
    max_gpus: Optional[int] = Field(default=None, ge=1, description="Maximum GPUs allowed")
    requires_nvlink: bool = Field(default=False, description="Requires NVLink connectivity")
    requires_tensor_cores: bool = Field(default=False, description="Requires tensor cores")
    
    @validator('max_gpus')
    def validate_max_gpus(cls, v, values):
        if v is not None and 'min_gpus' in values and v < values['min_gpus']:
            raise ValueError("max_gpus must be >= min_gpus")
        return v


class TrainingJob(BaseModel):
    """Production-grade training job specification"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Job name")
    model_type: str = Field(..., description="ML model type")
    dataset: str = Field(..., description="Training dataset")
    
    # Resource requirements
    requirements: WorkloadRequirements
    
    # Scheduling parameters
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1-10)")
    max_runtime_hours: Optional[int] = Field(default=None, gt=0)
    checkpoint_interval_minutes: int = Field(default=30, ge=1)
    
    # Fault tolerance settings
    enable_checkpointing: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0)
    fault_tolerance_level: str = Field(default="standard", regex="^(minimal|standard|aggressive)$")
    
    # Status tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Resource allocation (populated after scheduling)
    allocated_gpu_ids: List[str] = Field(default_factory=list)
    
    # Performance tracking
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class AllocationResult(BaseModel):
    """Result of resource allocation optimization"""
    
    strategy_used: OptimizationStrategy
    allocations: Dict[str, List[str]] = Field(..., description="task_id -> [gpu_ids]")
    
    # Performance metrics
    optimization_time_seconds: float = Field(..., ge=0.0)
    allocation_efficiency: float = Field(..., ge=0.0, le=1.0)
    resource_utilization: float = Field(..., ge=0.0, le=1.0)
    
    # Quality metrics
    tasks_allocated: int = Field(..., ge=0)
    total_tasks: int = Field(..., ge=0)
    gpu_utilization_distribution: Dict[str, float] = Field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cluster_size: int = Field(..., gt=0)
    
    @property
    def allocation_success_rate(self) -> float:
        """Calculate allocation success rate"""
        if self.total_tasks == 0:
            return 0.0
        return self.tasks_allocated / self.total_tasks
    
    class Config:
        validate_assignment = True
        use_enum_values = True


class ClusterConfiguration(BaseModel):
    """Enterprise cluster configuration"""
    
    cluster_id: str = Field(..., description="Unique cluster identifier")
    name: str = Field(..., description="Cluster name")
    
    # Optimization settings
    default_strategy: OptimizationStrategy = Field(default=OptimizationStrategy.HYBRID_OPTIMIZATION)
    enable_predictive_scaling: bool = Field(default=True)
    enable_fault_tolerance: bool = Field(default=True)
    
    # Performance tuning
    allocation_timeout_seconds: int = Field(default=30, gt=0)
    health_check_interval_seconds: int = Field(default=60, gt=0)
    metrics_collection_interval_seconds: int = Field(default=30, gt=0)
    
    # Fault tolerance
    failure_detection_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    recovery_timeout_seconds: int = Field(default=300, gt=0)
    max_concurrent_failures: int = Field(default=10, ge=1)
    
    # Resource limits
    max_gpu_utilization: float = Field(default=0.95, gt=0.0, le=1.0)
    memory_reservation_buffer: float = Field(default=0.1, ge=0.0, le=0.5)
    
    # Networking
    enable_nvlink_optimization: bool = Field(default=True)
    inter_node_bandwidth_gbps: float = Field(default=100.0, gt=0.0)
    
    class Config:
        validate_assignment = True
        use_enum_values = True


@dataclass
class PerformanceMetrics:
    """Performance measurement container"""
    
    # Timing metrics
    allocation_time_ms: float
    optimization_time_ms: float
    total_time_ms: float
    
    # Resource metrics
    gpu_utilization_avg: float
    gpu_utilization_std: float
    memory_utilization_avg: float
    memory_efficiency: float
    
    # Quality metrics
    allocation_success_rate: float
    load_balance_score: float
    fault_tolerance_score: float
    
    # Scalability metrics
    cluster_size: int
    tasks_processed: int
    throughput_tasks_per_second: float
    
    # Research validation metrics
    neocpu_improvement_factor: Optional[float] = None
    tvm_improvement_factor: Optional[float] = None
    hybrid_improvement_factor: Optional[float] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class OptimizationContext(BaseModel):
    """Context for optimization decisions"""
    
    cluster_config: ClusterConfiguration
    available_gpus: List[GPUNode]
    pending_tasks: List[TrainingJob]
    
    # Historical data
    historical_performance: Dict[str, List[float]] = Field(default_factory=dict)
    failure_patterns: Dict[str, int] = Field(default_factory=dict)
    
    # Real-time state
    current_allocations: Dict[str, List[str]] = Field(default_factory=dict)
    cluster_load: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Optimization constraints
    optimization_budget_seconds: float = Field(default=30.0, gt=0.0)
    quality_target: float = Field(default=0.9, gt=0.0, le=1.0)
    
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
