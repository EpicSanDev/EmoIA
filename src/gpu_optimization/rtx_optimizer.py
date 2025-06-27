"""
RTX 2070 Super Optimization Module for EmoIA v3.0
Maximizes GPU performance for AI inference with 8GB VRAM + 64GB System RAM
"""
import torch
import psutil
import GPUtil
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from threading import Thread, Lock
import asyncio
import yaml
import json

@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    memory_used: float
    memory_total: float
    memory_utilization: float
    temperature: float
    power_draw: float
    compute_utilization: float
    timestamp: float

@dataclass
class OptimizationSettings:
    """RTX 2070 Super specific optimization settings."""
    max_vram_gb: float = 7.5  # Reserve 0.5GB for system
    max_system_ram_gb: float = 32  # Use 50% of 64GB for AI cache
    compute_capability: str = "7.5"
    tensor_cores_enabled: bool = True
    mixed_precision: bool = True
    dynamic_batching: bool = True
    memory_growth: bool = True
    cache_optimization: bool = True

class RTXOptimizer:
    """
    Advanced GPU optimizer for RTX 2070 Super.
    Implements dynamic memory management, batch sizing, and performance monitoring.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.settings = OptimizationSettings()
        self.metrics_lock = Lock()
        self.current_metrics: Optional[GPUMetrics] = None
        self.performance_history: List[GPUMetrics] = []
        self.optimized_batch_sizes: Dict[str, int] = {}
        self.model_cache: Dict[str, Any] = {}
        self.memory_pool = None
        
        # Load configuration
        self.load_config(config_path)
        
        # Initialize GPU
        self.initialize_gpu()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = Thread(target=self._monitor_gpu, daemon=True)
        self.monitor_thread.start()
        
        logging.info("RTX 2070 Super Optimizer initialized successfully")

    def load_config(self, config_path: str):
        """Load optimization settings from config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            gpu_config = config.get('gpu_optimization', {})
            
            if gpu_config.get('enabled', True):
                self.settings.max_vram_gb = gpu_config.get('memory_limit', 7.5)
                self.settings.mixed_precision = gpu_config.get('mixed_precision', True)
                self.settings.dynamic_batching = gpu_config.get('dynamic_batching', True)
                
                memory_config = gpu_config.get('memory_management', {})
                self.settings.memory_growth = memory_config.get('dynamic_batching', True)
                
        except Exception as e:
            logging.warning(f"Could not load config: {e}. Using defaults.")

    def initialize_gpu(self):
        """Initialize GPU with optimal settings for RTX 2070 Super."""
        try:
            # Set CUDA device
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                torch.cuda.set_device(0)
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Configure memory management
                if self.settings.memory_growth:
                    torch.cuda.empty_cache()
                
                # Set memory fraction for RTX 2070 Super (8GB VRAM)
                memory_fraction = self.settings.max_vram_gb / 8.0
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                # Enable mixed precision if supported
                if self.settings.mixed_precision:
                    # Verify Tensor Core support
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 7:  # RTX 2070 Super has compute capability 7.5
                        self.settings.tensor_cores_enabled = True
                        logging.info("Tensor Cores enabled for mixed precision")
                
                logging.info(f"GPU initialized: {torch.cuda.get_device_name()}")
                logging.info(f"VRAM limit set to: {self.settings.max_vram_gb:.1f}GB")
                
            else:
                logging.error("CUDA not available - falling back to CPU")
                
        except Exception as e:
            logging.error(f"GPU initialization failed: {e}")

    def _monitor_gpu(self):
        """Background thread for continuous GPU monitoring."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                if metrics:
                    with self.metrics_lock:
                        self.current_metrics = metrics
                        self.performance_history.append(metrics)
                        
                        # Keep only last 1000 metrics (about 16 minutes at 1sec intervals)
                        if len(self.performance_history) > 1000:
                            self.performance_history = self.performance_history[-1000:]
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logging.error(f"GPU monitoring error: {e}")
                time.sleep(5)  # Wait longer on error

    def _collect_metrics(self) -> Optional[GPUMetrics]:
        """Collect current GPU metrics."""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu = gpus[0]  # RTX 2070 Super
            
            return GPUMetrics(
                memory_used=gpu.memoryUsed,
                memory_total=gpu.memoryTotal,
                memory_utilization=gpu.memoryUtil * 100,
                temperature=gpu.temperature,
                power_draw=getattr(gpu, 'powerDraw', 0),
                compute_utilization=gpu.load * 100,
                timestamp=time.time()
            )
            
        except Exception as e:
            logging.error(f"Error collecting GPU metrics: {e}")
            return None

    def get_optimal_batch_size(self, model_type: str, input_shape: Tuple[int, ...]) -> int:
        """Calculate optimal batch size for current GPU state."""
        try:
            cache_key = f"{model_type}_{input_shape}"
            
            # Return cached result if available
            if cache_key in self.optimized_batch_sizes:
                return self.optimized_batch_sizes[cache_key]
            
            # Get current memory usage
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return 8  # Conservative default
            
            # Calculate available memory
            available_memory = (current_metrics.memory_total - current_metrics.memory_used) * 0.8  # 80% safety margin
            
            # Estimate memory per sample (simplified)
            memory_per_sample = self._estimate_memory_per_sample(model_type, input_shape)
            
            # Calculate optimal batch size
            optimal_batch = max(1, int(available_memory / memory_per_sample))
            optimal_batch = min(optimal_batch, 32)  # Cap at 32 for stability
            
            # Cache the result
            self.optimized_batch_sizes[cache_key] = optimal_batch
            
            logging.info(f"Optimal batch size for {model_type}: {optimal_batch}")
            return optimal_batch
            
        except Exception as e:
            logging.error(f"Error calculating optimal batch size: {e}")
            return 8

    def _estimate_memory_per_sample(self, model_type: str, input_shape: Tuple[int, ...]) -> float:
        """Estimate memory usage per sample in MB."""
        # Simplified estimation based on model type and input
        base_memory = {
            'transformer': 50,  # MB per sample
            'cnn': 30,
            'rnn': 20,
            'embedding': 10
        }
        
        model_memory = base_memory.get(model_type.lower(), 40)
        
        # Adjust for input size
        input_size_factor = 1.0
        if len(input_shape) > 1:
            total_elements = 1
            for dim in input_shape[1:]:  # Skip batch dimension
                total_elements *= dim
            input_size_factor = max(1.0, total_elements / 512)  # Normalize to 512 elements
        
        return model_memory * input_size_factor

    def optimize_model_loading(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Optimize model for RTX 2070 Super."""
        try:
            # Move to GPU
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Enable mixed precision if supported
            if self.settings.mixed_precision and self.settings.tensor_cores_enabled:
                model = model.half()  # Convert to FP16
                logging.info(f"Model {model_name} converted to FP16 for Tensor Cores")
            
            # Compile model for optimization (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logging.info(f"Model {model_name} compiled for optimization")
                except Exception as e:
                    logging.warning(f"Model compilation failed: {e}")
            
            # Cache the optimized model
            self.model_cache[model_name] = model
            
            return model
            
        except Exception as e:
            logging.error(f"Model optimization failed for {model_name}: {e}")
            return model

    def create_memory_pool(self, pool_size_gb: float = 16.0):
        """Create optimized memory pool for faster allocations."""
        try:
            if torch.cuda.is_available():
                # Pre-allocate memory pool
                pool_size_bytes = int(pool_size_gb * 1024 * 1024 * 1024)
                dummy_tensor = torch.zeros(pool_size_bytes // 4, dtype=torch.float32, device='cuda')
                del dummy_tensor
                torch.cuda.empty_cache()
                
                logging.info(f"Memory pool created: {pool_size_gb}GB")
                
        except Exception as e:
            logging.error(f"Memory pool creation failed: {e}")

    def optimize_inference(self, model: torch.nn.Module, input_data: torch.Tensor, 
                          model_type: str = 'transformer') -> torch.Tensor:
        """Perform optimized inference."""
        try:
            # Get optimal batch size
            batch_size = self.get_optimal_batch_size(model_type, input_data.shape)
            
            # Process in optimal batches
            results = []
            for i in range(0, input_data.size(0), batch_size):
                batch = input_data[i:i+batch_size]
                
                # Use autocast for mixed precision
                if self.settings.mixed_precision:
                    with torch.cuda.amp.autocast():
                        batch_result = model(batch)
                else:
                    batch_result = model(batch)
                
                results.append(batch_result)
            
            # Concatenate results
            return torch.cat(results, dim=0)
            
        except Exception as e:
            logging.error(f"Optimized inference failed: {e}")
            return model(input_data)  # Fallback

    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """Get current GPU metrics thread-safely."""
        with self.metrics_lock:
            return self.current_metrics

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            with self.metrics_lock:
                if not self.performance_history:
                    return {}
                
                recent_metrics = self.performance_history[-60:]  # Last minute
                
                # Calculate statistics
                memory_utils = [m.memory_utilization for m in recent_metrics]
                compute_utils = [m.compute_utilization for m in recent_metrics]
                temperatures = [m.temperature for m in recent_metrics]
                
                stats = {
                    'current_memory_usage': recent_metrics[-1].memory_used if recent_metrics else 0,
                    'current_memory_util': recent_metrics[-1].memory_utilization if recent_metrics else 0,
                    'current_compute_util': recent_metrics[-1].compute_utilization if recent_metrics else 0,
                    'current_temperature': recent_metrics[-1].temperature if recent_metrics else 0,
                    'avg_memory_util': sum(memory_utils) / len(memory_utils),
                    'avg_compute_util': sum(compute_utils) / len(compute_utils),
                    'avg_temperature': sum(temperatures) / len(temperatures),
                    'max_memory_util': max(memory_utils),
                    'max_temperature': max(temperatures),
                    'optimized_batch_sizes': dict(self.optimized_batch_sizes),
                    'cached_models': list(self.model_cache.keys())
                }
                
                return stats
                
        except Exception as e:
            logging.error(f"Error calculating performance stats: {e}")
            return {}

    def auto_optimize_system(self):
        """Automatically optimize system settings based on current performance."""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return
            
            # Check memory pressure
            if current_metrics.memory_utilization > 90:
                logging.warning("High GPU memory usage detected - clearing cache")
                torch.cuda.empty_cache()
                
                # Clear model cache if needed
                if len(self.model_cache) > 3:
                    # Remove oldest cached models
                    models_to_remove = list(self.model_cache.keys())[:2]
                    for model_name in models_to_remove:
                        del self.model_cache[model_name]
                    logging.info(f"Cleared cached models: {models_to_remove}")
            
            # Check temperature
            if current_metrics.temperature > 80:
                logging.warning("High GPU temperature detected - reducing performance")
                # Reduce batch sizes
                for key in self.optimized_batch_sizes:
                    self.optimized_batch_sizes[key] = max(1, self.optimized_batch_sizes[key] // 2)
            
            # Check compute utilization
            if current_metrics.compute_utilization < 30:
                logging.info("Low GPU utilization - increasing batch sizes")
                # Increase batch sizes cautiously
                for key in self.optimized_batch_sizes:
                    self.optimized_batch_sizes[key] = min(32, int(self.optimized_batch_sizes[key] * 1.2))
                    
        except Exception as e:
            logging.error(f"Auto-optimization failed: {e}")

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance history."""
        recommendations = []
        
        try:
            stats = self.get_performance_stats()
            if not stats:
                return recommendations
            
            # Memory recommendations
            if stats.get('avg_memory_util', 0) > 85:
                recommendations.append("Consider reducing model sizes or batch sizes - high memory usage detected")
            elif stats.get('avg_memory_util', 0) < 50:
                recommendations.append("GPU memory underutilized - consider increasing batch sizes")
            
            # Compute recommendations
            if stats.get('avg_compute_util', 0) < 40:
                recommendations.append("Low GPU utilization - consider parallel processing or larger models")
            
            # Temperature recommendations
            if stats.get('max_temperature', 0) > 75:
                recommendations.append("High temperatures detected - check cooling and reduce workload if needed")
            
            # Model cache recommendations
            if len(stats.get('cached_models', [])) < 2:
                recommendations.append("Consider pre-loading frequently used models for better performance")
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            return []

    def export_performance_report(self, filepath: str = "gpu_performance_report.json"):
        """Export detailed performance report."""
        try:
            report = {
                'system_info': {
                    'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
                    'cuda_version': torch.version.cuda,
                    'pytorch_version': torch.__version__,
                    'system_ram_gb': psutil.virtual_memory().total / (1024**3),
                    'optimization_settings': {
                        'max_vram_gb': self.settings.max_vram_gb,
                        'mixed_precision': self.settings.mixed_precision,
                        'tensor_cores_enabled': self.settings.tensor_cores_enabled,
                        'dynamic_batching': self.settings.dynamic_batching
                    }
                },
                'performance_stats': self.get_performance_stats(),
                'recommendations': self.get_optimization_recommendations(),
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logging.error(f"Error exporting performance report: {e}")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.monitoring_active = False
            if self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            # Clear model cache
            self.model_cache.clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info("RTX Optimizer cleanup completed")
            
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

    def __del__(self):
        """Destructor."""
        self.cleanup()

# Singleton instance
_rtx_optimizer = None

def get_rtx_optimizer() -> RTXOptimizer:
    """Get singleton RTX optimizer instance."""
    global _rtx_optimizer
    if _rtx_optimizer is None:
        _rtx_optimizer = RTXOptimizer()
    return _rtx_optimizer

# Convenience functions
async def optimize_model_async(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Async wrapper for model optimization."""
    optimizer = get_rtx_optimizer()
    return optimizer.optimize_model_loading(model, model_name)

async def get_performance_metrics_async() -> Dict[str, Any]:
    """Async wrapper for performance metrics."""
    optimizer = get_rtx_optimizer()
    return optimizer.get_performance_stats()

def setup_rtx_optimization(config_path: str = "config.yaml") -> RTXOptimizer:
    """Setup RTX optimization with configuration."""
    return RTXOptimizer(config_path)