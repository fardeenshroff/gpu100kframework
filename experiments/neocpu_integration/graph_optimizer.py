"""
Simple GPU Cluster Graph Optimization Experiment
Demonstrates core concepts from NeoCPU research
Author: Fardeen Fayyaz Shroff
"""

import time
import logging
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class GPUNode:
    """Represents a GPU in our cluster"""
    id: str
    memory_gb: int
    compute_power: float  # TFLOPS
    current_load: float = 0.0  # 0.0 to 1.0

@dataclass
class WorkloadTask:
    """Represents a training job that needs GPU resources"""
    id: str
    model_name: str
    memory_needed: int
    compute_needed: float
    priority: int  # 1-10, higher = more important

class SimpleGraphOptimizer:
    """
    Simplified version of graph optimization inspired by NeoCPU research
    Demonstrates the core concepts without overwhelming complexity
    """
    
    def __init__(self, gpus: List[GPUNode]):
        self.gpus = {gpu.id: gpu for gpu in gpus}
        print(f"🚀 Initialized cluster with {len(gpus)} GPUs")
    
    def baseline_allocation(self, tasks: List[WorkloadTask]) -> Dict[str, str]:
        """
        Baseline method: Simple first-fit allocation
        This is what we'll compare our optimized version against
        """
        print("📊 Running baseline allocation...")
        start_time = time.time()
        
        allocations = {}  # task_id -> gpu_id
        
        # Sort tasks by priority (high to low)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            allocated = False
            for gpu_id, gpu in self.gpus.items():
                # Check if GPU has enough resources
                if (gpu.memory_gb >= task.memory_needed and 
                    gpu.compute_power >= task.compute_needed and
                    gpu.current_load < 0.8):  # Keep some headroom
                    
                    allocations[task.id] = gpu_id
                    gpu.current_load += 0.2  # Simplified load calculation
                    allocated = True
                    break
            
            if not allocated:
                print(f"⚠️  Could not allocate task {task.id}")
        
        duration = time.time() - start_time
        print(f"✅ Baseline allocation completed in {duration:.3f}s")
        print(f"📈 Allocated {len(allocations)}/{len(tasks)} tasks")
        
        return allocations
    
    def optimized_allocation(self, tasks: List[WorkloadTask]) -> Dict[str, str]:
        """
        Optimized method: Inspired by NeoCPU's two-stage optimization
        Stage 1: Local optimization
        Stage 2: Global coordination
        """
        print("🧠 Running optimized allocation (NeoCPU-inspired)...")
        start_time = time.time()
        
        allocations = {}
        
        # Stage 1: Create GPU groups (simplified)
        gpu_groups = self._create_gpu_groups()
        
        # Stage 2: Optimize within each group
        for group_name, gpu_ids in gpu_groups.items():
            print(f"🔧 Optimizing group {group_name} ({len(gpu_ids)} GPUs)")
            group_allocations = self._optimize_group(gpu_ids, tasks)
            allocations.update(group_allocations)
        
        # Stage 3: Global rebalancing
        allocations = self._global_rebalance(allocations, tasks)
        
        duration = time.time() - start_time
        print(f"✅ Optimized allocation completed in {duration:.3f}s")
        print(f"📈 Allocated {len(allocations)}/{len(tasks)} tasks")
        
        return allocations
    
    def _create_gpu_groups(self) -> Dict[str, List[str]]:
        """Create logical GPU groups for optimization"""
        gpu_ids = list(self.gpus.keys())
        groups = {}
        
        # Simple grouping: 4 GPUs per group
        group_size = 4
        for i in range(0, len(gpu_ids), group_size):
            group_name = f"group_{i//group_size + 1}"
            groups[group_name] = gpu_ids[i:i+group_size]
        
        return groups
    
    def _optimize_group(self, gpu_ids: List[str], tasks: List[WorkloadTask]) -> Dict[str, str]:
        """Optimize allocation within a GPU group"""
        local_allocations = {}
        
        # Focus on high-priority tasks first
        high_priority_tasks = [t for t in tasks if t.priority >= 7]
        
        for task in high_priority_tasks:
            if task.id in local_allocations:
                continue  # Already allocated
                
            best_gpu = self._find_best_gpu_in_group(task, gpu_ids)
            if best_gpu:
                local_allocations[task.id] = best_gpu
                self.gpus[best_gpu].current_load += 0.15
        
        return local_allocations
    
    def _find_best_gpu_in_group(self, task: WorkloadTask, gpu_ids: List[str]) -> str:
        """Find the best GPU for a task within a group"""
        best_gpu = None
        best_score = float('inf')
        
        for gpu_id in gpu_ids:
            gpu = self.gpus[gpu_id]
            
            # Check if GPU can handle the task
            if (gpu.memory_gb < task.memory_needed or 
                gpu.compute_power < task.compute_needed or
                gpu.current_load > 0.9):
                continue
            
            # Calculate efficiency score (lower is better)
            memory_efficiency = task.memory_needed / gpu.memory_gb
            compute_efficiency = task.compute_needed / gpu.compute_power
            load_penalty = gpu.current_load
            
            score = memory_efficiency + compute_efficiency + load_penalty
            
            if score < best_score:
                best_score = score
                best_gpu = gpu_id
        
        return best_gpu
    
    def _global_rebalance(self, allocations: Dict[str, str], tasks: List[WorkloadTask]) -> Dict[str, str]:
        """Global rebalancing to improve overall efficiency"""
        print("🔄 Performing global rebalancing...")
        
        # Handle unallocated tasks
        allocated_task_ids = set(allocations.keys())
        all_task_ids = {task.id for task in tasks}
        unallocated_task_ids = all_task_ids - allocated_task_ids
        
        # Try to allocate remaining tasks
        for task in tasks:
            if task.id in unallocated_task_ids:
                best_gpu = self._find_best_gpu_globally(task)
                if best_gpu:
                    allocations[task.id] = best_gpu
                    self.gpus[best_gpu].current_load += 0.1
        
        return allocations
    
    def _find_best_gpu_globally(self, task: WorkloadTask) -> str:
        """Find the best GPU across the entire cluster"""
        best_gpu = None
        best_score = float('inf')
        
        for gpu_id, gpu in self.gpus.items():
            if (gpu.memory_gb >= task.memory_needed and 
                gpu.compute_power >= task.compute_needed and
                gpu.current_load < 0.95):
                
                # Prefer less loaded GPUs
                score = gpu.current_load + (task.memory_needed / gpu.memory_gb)
                
                if score < best_score:
                    best_score = score
                    best_gpu = gpu_id
        
        return best_gpu
    
    def compare_methods(self, tasks: List[WorkloadTask]):
        """Compare baseline vs optimized allocation methods"""
        print("🆚 Comparing allocation methods...")
        print("=" * 50)
        
        # Reset GPU loads for fair comparison
        for gpu in self.gpus.values():
            gpu.current_load = 0.0
        
        # Test baseline method
        baseline_result = self.baseline_allocation(tasks)
        baseline_efficiency = self._calculate_efficiency(baseline_result, tasks)
        
        # Reset GPU loads
        for gpu in self.gpus.values():
            gpu.current_load = 0.0
        
        # Test optimized method
        optimized_result = self.optimized_allocation(tasks)
        optimized_efficiency = self._calculate_efficiency(optimized_result, tasks)
        
        # Print comparison
        print("\n📊 RESULTS COMPARISON:")
        print(f"Baseline method:")
        print(f"  - Tasks allocated: {len(baseline_result)}/{len(tasks)}")
        print(f"  - Efficiency score: {baseline_efficiency:.3f}")
        
        print(f"Optimized method:")
        print(f"  - Tasks allocated: {len(optimized_result)}/{len(tasks)}")
        print(f"  - Efficiency score: {optimized_efficiency:.3f}")
        
        if optimized_efficiency > baseline_efficiency:
            improvement = ((optimized_efficiency - baseline_efficiency) / baseline_efficiency) * 100
            print(f"🎉 Improvement: {improvement:.1f}% better efficiency!")
        
        return baseline_result, optimized_result
    
    def _calculate_efficiency(self, allocations: Dict[str, str], tasks: List[WorkloadTask]) -> float:
        """Calculate overall allocation efficiency"""
        if not allocations:
            return 0.0
        
        total_efficiency = 0.0
        task_dict = {task.id: task for task in tasks}
        
        for task_id, gpu_id in allocations.items():
            task = task_dict[task_id]
            gpu = self.gpus[gpu_id]
            
            # Calculate resource utilization efficiency
            memory_util = task.memory_needed / gpu.memory_gb
            compute_util = task.compute_needed / gpu.compute_power
            
            # Higher utilization = better efficiency
            efficiency = (memory_util + compute_util) / 2
            total_efficiency += efficiency
        
        return total_efficiency / len(allocations)


# Demo function to test the optimizer
def run_experiment():
    """Run a simple experiment to demonstrate the optimizer"""
    print("🚀 Starting GPU Cluster Optimization Experiment")
    print("=" * 60)
    
    # Create sample GPU cluster
    gpus = [
        GPUNode("gpu_1", memory_gb=32, compute_power=15.0),
        GPUNode("gpu_2", memory_gb=32, compute_power=15.0),
        GPUNode("gpu_3", memory_gb=24, compute_power=12.0),
        GPUNode("gpu_4", memory_gb=24, compute_power=12.0),
        GPUNode("gpu_5", memory_gb=16, compute_power=8.0),
        GPUNode("gpu_6", memory_gb=16, compute_power=8.0),
        GPUNode("gpu_7", memory_gb=32, compute_power=20.0),
        GPUNode("gpu_8", memory_gb=32, compute_power=20.0),
    ]
    
    # Create sample workload tasks
    tasks = [
        WorkloadTask("task_1", "ResNet-50", memory_needed=16, compute_needed=8.0, priority=9),
        WorkloadTask("task_2", "BERT-Large", memory_needed=24, compute_needed=12.0, priority=8),
        WorkloadTask("task_3", "GPT-3", memory_needed=28, compute_needed=18.0, priority=10),
        WorkloadTask("task_4", "VGG-19", memory_needed=12, compute_needed=6.0, priority=7),
        WorkloadTask("task_5", "Inception-v3", memory_needed=14, compute_needed=7.0, priority=6),
        WorkloadTask("task_6", "MobileNet", memory_needed=8, compute_needed=4.0, priority=5),
        WorkloadTask("task_7", "AlexNet", memory_needed=10, compute_needed=5.0, priority=4),
    ]
    
    # Initialize optimizer
    optimizer = SimpleGraphOptimizer(gpus)
    
    # Run comparison
    baseline_result, optimized_result = optimizer.compare_methods(tasks)
    
    print("\n🎯 Experiment completed successfully!")
    return baseline_result, optimized_result


if __name__ == "__main__":
    run_experiment()

