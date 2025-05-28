#!/usr/bin/env python3
"""
Performance Health Monitor for AutoTuneSyncer
Monitors system resources, processing performance, and provides insights
"""

import psutil
import time
import json
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import subprocess

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    active_processes: int = 0
    ffmpeg_processes: int = 0
    python_processes: int = 0

@dataclass
class ProcessingMetrics:
    """Video processing performance metrics"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_videos: int = 0
    processed_videos: int = 0
    failed_videos: int = 0
    average_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    total_processing_time: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class PerformanceHealthMonitor:
    """Monitor system health and processing performance"""
    
    def __init__(self, monitoring_interval=5):
        self.monitoring_interval = monitoring_interval
        self.monitoring = False
        self.monitor_thread = None
        self.system_metrics = []
        self.processing_sessions = {}
        self.current_session = None
        
        # Setup logging
        self.setup_logging()
        
        # Check GPU availability
        self.gpu_available = self.check_gpu_availability()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'health_monitor.log'),
                logging.StreamHandler()
            ]        )
        self.logger = logging.getLogger(__name__)
    
    def check_gpu_availability(self):
        """Check if GPU monitoring is available"""
        try:
            # First try our enhanced GPU setup
            from python.gpu_setup import gpu_available, torch_cuda_available
            if gpu_available and torch_cuda_available:
                return True
        except ImportError:
            pass
            
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_gpu_metrics(self):
        """Get GPU metrics if available"""
        if not self.gpu_available:
            return None, None, None
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Take first GPU
                    values = lines[0].split(', ')
                    memory_used = float(values[0])
                    memory_total = float(values[1])
                    utilization = float(values[2])
                    return memory_used, memory_total, utilization
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
        
        return None, None, None
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics
        gpu_memory_used, gpu_memory_total, gpu_utilization = self.get_gpu_metrics()
        
        # Process counts
        active_processes = len(psutil.pids())
        ffmpeg_processes = 0
        python_processes = 0
        
        for proc in psutil.process_iter(['name']):
            try:
                name = proc.info['name'].lower()
                if 'ffmpeg' in name:
                    ffmpeg_processes += 1
                elif 'python' in name:
                    python_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.used / disk.total * 100,
            disk_free_gb=disk.free / (1024**3),
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            active_processes=active_processes,
            ffmpeg_processes=ffmpeg_processes,
            python_processes=python_processes
        )
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Keep only last 1000 measurements to avoid memory issues
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                # Check for performance issues
                self._check_performance_alerts(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)  # Longer delay on error
    
    def _check_performance_alerts(self, metrics: SystemMetrics):
        """Check for performance issues and log alerts"""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_free_gb < 1.0:
            alerts.append(f"Low disk space: {metrics.disk_free_gb:.1f}GB free")
        
        if metrics.gpu_utilization and metrics.gpu_utilization > 95:
            alerts.append(f"High GPU utilization: {metrics.gpu_utilization:.1f}%")
        
        if metrics.ffmpeg_processes > 10:
            alerts.append(f"Many FFmpeg processes: {metrics.ffmpeg_processes}")
        
        for alert in alerts:
            self.logger.warning(f"PERFORMANCE ALERT: {alert}")
    
    def start_processing_session(self, session_id: str, total_videos: int = 0):
        """Start tracking a processing session"""
        session = ProcessingMetrics(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            total_videos=total_videos
        )
        
        self.processing_sessions[session_id] = session
        self.current_session = session_id
        self.logger.info(f"Started processing session: {session_id}")
        
        return session
    
    def update_processing_progress(self, session_id: str, processed: int = 0, failed: int = 0):
        """Update processing progress"""
        if session_id in self.processing_sessions:
            session = self.processing_sessions[session_id]
            session.processed_videos = processed
            session.failed_videos = failed
            
            if self.system_metrics:
                latest = self.system_metrics[-1]
                session.peak_memory_usage = max(session.peak_memory_usage, latest.memory_percent)
                session.peak_cpu_usage = max(session.peak_cpu_usage, latest.cpu_percent)
    
    def end_processing_session(self, session_id: str):
        """End a processing session"""
        if session_id in self.processing_sessions:
            session = self.processing_sessions[session_id]
            session.end_time = datetime.now().isoformat()
            
            # Calculate total processing time
            start_time = datetime.fromisoformat(session.start_time)
            end_time = datetime.fromisoformat(session.end_time)
            session.total_processing_time = (end_time - start_time).total_seconds()
            
            # Calculate average processing time per video
            if session.processed_videos > 0:
                session.average_processing_time = session.total_processing_time / session.processed_videos
            
            self.logger.info(f"Completed processing session: {session_id}")
            
            if self.current_session == session_id:
                self.current_session = None
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.system_metrics:
            return {"error": "No metrics collected"}
        
        # Calculate system metrics summary
        cpu_values = [m.cpu_percent for m in self.system_metrics]
        memory_values = [m.memory_percent for m in self.system_metrics]
        
        system_summary = {
            "monitoring_duration_minutes": len(self.system_metrics) * self.monitoring_interval / 60,
            "samples_collected": len(self.system_metrics),
            "cpu": {
                "average": sum(cpu_values) / len(cpu_values),
                "peak": max(cpu_values),
                "current": cpu_values[-1] if cpu_values else 0
            },
            "memory": {
                "average": sum(memory_values) / len(memory_values),
                "peak": max(memory_values),
                "current": memory_values[-1] if memory_values else 0
            },
            "latest_metrics": asdict(self.system_metrics[-1])
        }
        
        # Add GPU summary if available
        if self.gpu_available and self.system_metrics[-1].gpu_utilization is not None:
            gpu_util_values = [m.gpu_utilization for m in self.system_metrics if m.gpu_utilization is not None]
            if gpu_util_values:
                system_summary["gpu"] = {
                    "average_utilization": sum(gpu_util_values) / len(gpu_util_values),
                    "peak_utilization": max(gpu_util_values),
                    "current_memory_used_mb": self.system_metrics[-1].gpu_memory_used,
                    "total_memory_mb": self.system_metrics[-1].gpu_memory_total
                }
        
        # Processing sessions summary
        sessions_summary = {}
        for session_id, session in self.processing_sessions.items():
            sessions_summary[session_id] = asdict(session)
        
        return {
            "system": system_summary,
            "processing_sessions": sessions_summary,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self):
        """Generate performance recommendations"""
        recommendations = []
        
        if not self.system_metrics:
            return recommendations
        
        latest = self.system_metrics[-1]
        
        # CPU recommendations
        if latest.cpu_percent > 80:
            recommendations.append("Consider reducing parallel processing workers to decrease CPU load")
        
        # Memory recommendations
        if latest.memory_percent > 80:
            recommendations.append("Consider reducing memory-intensive operations or increasing system RAM")
        
        # Disk recommendations
        if latest.disk_free_gb < 5:
            recommendations.append("Free up disk space - low storage may impact processing performance")
        
        # GPU recommendations
        if latest.gpu_utilization and latest.gpu_utilization < 50:
            recommendations.append("GPU utilization is low - consider enabling hardware acceleration")
        
        # Process recommendations
        if latest.ffmpeg_processes > 8:
            recommendations.append("Many FFmpeg processes detected - consider reducing concurrent operations")
        
        return recommendations
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON file"""
        data = {
            "export_time": datetime.now().isoformat(),
            "monitoring_interval": self.monitoring_interval,
            "gpu_available": self.gpu_available,
            "system_metrics": [asdict(m) for m in self.system_metrics],
            "processing_sessions": {k: asdict(v) for k, v in self.processing_sessions.items()},
            "summary": self.get_performance_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to: {output_path}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='AutoTuneSyncer Performance Health Monitor')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds')
    parser.add_argument('--export', help='Export metrics to JSON file')
    parser.add_argument('--session-id', help='Track a specific processing session')
    parser.add_argument('--daemon', action='store_true', help='Run as background daemon')
    
    args = parser.parse_args()
    
    monitor = PerformanceHealthMonitor(monitoring_interval=args.interval)
    
    try:
        monitor.start_monitoring()
        
        if args.session_id:
            monitor.start_processing_session(args.session_id)
        
        if args.daemon:
            print("Running in daemon mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(10)
        else:
            print(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    
    finally:
        if args.session_id:
            monitor.end_processing_session(args.session_id)
        
        monitor.stop_monitoring()
        
        # Print summary
        summary = monitor.get_performance_summary()
        print("\nPerformance Summary:")
        print(json.dumps(summary, indent=2))
        
        # Export if requested
        if args.export:
            monitor.export_metrics(args.export)

if __name__ == "__main__":
    main()
