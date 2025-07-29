#!/usr/bin/env python3
"""
Experiment Runner for GPU 100K Framework Research Integration
Author: Fardeen Fayyaz Shroff
"""

import sys
import os
import importlib.util
from pathlib import Path

def run_graph_optimization_experiment():
    """Run the NeoCPU graph optimization experiment"""
    print("🔬 Running Graph Optimization Experiment")
    print("=" * 50)
    
    try:
        # Import and run the experiment
        from neocpu_integration.graph_optimizer import run_experiment
        result = run_experiment()
        print("✅ Graph optimization experiment completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Graph optimization experiment failed: {e}")
        return False

def run_all_experiments():
    """Run all available experiments"""
    print("🚀 Running All GPU 100K Framework Experiments")
    print("=" * 60)
    
    experiments = [
        ("Graph Optimization", run_graph_optimization_experiment),
        # We'll add more experiments here later
    ]
    
    results = {}
    
    for name, experiment_func in experiments:
        print(f"\n📊 Starting: {name}")
        try:
            success = experiment_func()
            results[name] = "SUCCESS" if success else "FAILED"
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results[name] = "FAILED"
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for name, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {name}: {status}")
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    print(f"\n🎯 Overall: {success_count}/{total_count} experiments successful")
    
    if success_count == total_count:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check the output above for details.")
    
    return results

def show_available_experiments():
    """Show all available experiments"""
    print("📋 Available Experiments:")
    print("=" * 30)
    print("1. Graph Optimization (NeoCPU-inspired)")
    print("   - Two-stage optimization algorithm")
    print("   - Baseline vs optimized comparison")
    print("   - Resource allocation efficiency")
    print()
    print("🚧 Coming Soon:")
    print("2. TVM Layer Analysis (Vulnerability prediction)")
    print("3. Performance Benchmarking")
    print("4. Integration Testing")

def main():
    """Main experiment runner"""
    if len(sys.argv) < 2:
        print("🔬 GPU 100K Framework - Research Integration Experiments")
        print("=" * 55)
        print("Usage:")
        print("  python run_experiments.py [command]")
        print()
        print("Commands:")
        print("  list     - Show available experiments")
        print("  graph    - Run graph optimization experiment")
        print("  all      - Run all experiments")
        print()
        print("Examples:")
        print("  python run_experiments.py graph")
        print("  python run_experiments.py all")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        show_available_experiments()
    elif command == "graph":
        run_graph_optimization_experiment()
    elif command == "all":
        run_all_experiments()
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python run_experiments.py' to see available commands")

if __name__ == "__main__":
    # Add experiments directory to Python path
    experiments_dir = Path(__file__).parent
    sys.path.insert(0, str(experiments_dir))
    
    main()

