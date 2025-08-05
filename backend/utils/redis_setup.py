#!/usr/bin/env python3
"""
Redis Configuration Setup for AutoTuneSyncer
Initializes Redis cache service and creates default configuration
"""

import os
import json
import logging
import subprocess
import sys
from pathlib import Path

# Redis configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "db": int(os.getenv("REDIS_DB", 0)),
    "decode_responses": True,
    "socket_timeout": 30,
    "socket_connect_timeout": 30,
    "retry_on_timeout": True,
    "health_check_interval": 30
}

def check_redis_availability():
    """Check if Redis server is available"""
    try:
        import redis
        r = redis.Redis(**REDIS_CONFIG)
        r.ping()
        logging.info("Redis server is available")
        return True
    except Exception as e:
        logging.warning(f"Redis server not available: {e}")
        return False

def install_redis_locally():
    """Install Redis locally using package manager"""
    logging.info("Attempting to install Redis locally...")
    
    # Try different installation methods
    install_commands = [
        ["choco", "install", "redis-64"],  # Windows Chocolatey
        ["winget", "install", "Redis.Redis"],  # Windows winget
        ["brew", "install", "redis"],  # macOS Homebrew
        ["sudo", "apt-get", "install", "-y", "redis-server"],  # Ubuntu/Debian
        ["sudo", "yum", "install", "-y", "redis"],  # CentOS/RHEL
    ]
    
    for cmd in install_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logging.info(f"Redis installed successfully using: {' '.join(cmd)}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    logging.error("Could not install Redis automatically")
    return False

def start_redis_service():
    """Start Redis service"""
    logging.info("Starting Redis service...")
    
    start_commands = [
        ["redis-server"],  # Direct command
        ["sudo", "systemctl", "start", "redis"],  # systemd
        ["sudo", "service", "redis-server", "start"],  # SysV
        ["brew", "services", "start", "redis"],  # macOS Homebrew
    ]
    
    for cmd in start_commands:
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
            # Check if Redis is now available
            if check_redis_availability():
                logging.info(f"Redis service started successfully using: {' '.join(cmd)}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    logging.error("Could not start Redis service")
    return False

def create_redis_config_file():
    """Create Redis configuration file"""
    config_content = f"""# Redis Configuration for AutoTuneSyncer
# Performance-optimized settings

# Network
bind 127.0.0.1
port {REDIS_CONFIG['port']}
timeout 0
tcp-keepalive 300

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence for development (disable for production if not needed)
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile ""

# Performance
tcp-backlog 511
databases 16

# Disable some features for better performance
# appendonly no
# appendfsync everysec

# Security (uncomment and set password if needed)
# requirepass {REDIS_CONFIG['password']}
"""
    
    config_path = Path.cwd() / "redis.conf"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logging.info(f"Redis configuration file created at: {config_path}")
    return str(config_path)

def setup_redis_cache():
    """Setup Redis cache for AutoTuneSyncer"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Setting up Redis cache for AutoTuneSyncer...")
    
    # Check if Redis is already available
    if check_redis_availability():
        logging.info("Redis is already configured and running")
        return True
    
    # Try to install Redis if not available
    if not install_redis_locally():
        logging.warning("Could not install Redis. Cache will be disabled.")
        return False
    
    # Try to start Redis service
    if not start_redis_service():
        logging.warning("Could not start Redis service. Cache will be disabled.")
        return False
    
    # Create configuration file
    config_path = create_redis_config_file()
    
    # Final check
    if check_redis_availability():
        logging.info("Redis cache setup completed successfully")
        
        # Create cache setup info file
        setup_info = {
            "redis_config": REDIS_CONFIG,
            "config_file": config_path,
            "status": "configured",
            "setup_time": str(os.path.getmtime(config_path))
        }
        
        info_path = Path.cwd() / "redis_setup.json"
        with open(info_path, 'w') as f:
            json.dump(setup_info, f, indent=2)
        
        return True
    else:
        logging.error("Redis setup failed")
        return False

def get_cache_stats():
    """Get Redis cache statistics"""
    try:
        import redis
        r = redis.Redis(**REDIS_CONFIG)
        info = r.info()
        
        stats = {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0)
        }
        
        # Calculate hit rate
        hits = stats["keyspace_hits"]
        misses = stats["keyspace_misses"]
        if hits + misses > 0:
            stats["hit_rate"] = (hits / (hits + misses)) * 100
        else:
            stats["hit_rate"] = 0
        
        return stats
    except Exception as e:
        logging.error(f"Could not get cache stats: {e}")
        return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Redis Cache Setup for AutoTuneSyncer")
    parser.add_argument("--setup", action="store_true", help="Setup Redis cache")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--check", action="store_true", help="Check Redis availability")
    
    args = parser.parse_args()
    
    if args.setup:
        success = setup_redis_cache()
        sys.exit(0 if success else 1)
    elif args.stats:
        stats = get_cache_stats()
        print(json.dumps(stats, indent=2))
    elif args.check:
        available = check_redis_availability()
        print(f"Redis available: {available}")
        sys.exit(0 if available else 1)
    else:
        parser.print_help()
