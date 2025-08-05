import Redis from 'ioredis';
import process from 'process';

class CacheService {
  constructor() {
    // Load configuration from environment variables
    const redisConfig = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT) || 6379,
      password: process.env.REDIS_PASSWORD || undefined,
      db: parseInt(process.env.REDIS_DB) || 0,
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100,
      enableReadyCheck: false,
      lazyConnect: true,
      connectTimeout: 10000,
      commandTimeout: 5000,
      family: 4, // IPv4
    };

    // Add password only if provided
    if (process.env.REDIS_PASSWORD) {
      redisConfig.password = process.env.REDIS_PASSWORD;
    }

    this.redis = new Redis(redisConfig);
    this.fallbackCache = new Map(); // In-memory fallback cache
    this.isRedisAvailable = false;
    this.maxFallbackSize = 1000; // Limit fallback cache size

    this.redis.on('error', (err) => {
      console.warn(
        'Redis connection error, falling back to in-memory cache:',
        err.message
      );
      this.isRedisAvailable = false;
    });

    this.redis.on('connect', () => {
      console.log('Connected to Redis cache');
      this.isRedisAvailable = true;
    });

    this.redis.on('ready', () => {
      console.log('Redis cache is ready');
      this.isRedisAvailable = true;
    });

    this.redis.on('close', () => {
      console.warn('Redis connection closed, using fallback cache');
      this.isRedisAvailable = false;
    });

    // Health check interval
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, parseInt(process.env.HEALTH_CHECK_INTERVAL) || 30000);
  }

  async checkHealth() {
    try {
      if (this.redis.status === 'ready') {
        await this.redis.ping();
        this.isRedisAvailable = true;
      }
    } catch {
      this.isRedisAvailable = false;
    }
  }

  async get(key) {
    try {
      if (this.isRedisAvailable) {
        const value = await this.redis.get(key);
        return value ? JSON.parse(value) : null;
      } else {
        // Fallback to in-memory cache
        return this.fallbackCache.get(key) || null;
      }
    } catch (error) {
      console.warn('Cache get error:', error.message);
      return this.fallbackCache.get(key) || null;
    }
  }

  async set(key, value, ttl = 3600) {
    try {
      const serialized = JSON.stringify(value);

      if (this.isRedisAvailable) {
        await this.redis.setex(key, ttl, serialized);
      } else {
        // Fallback to in-memory cache with size limit
        if (this.fallbackCache.size >= this.maxFallbackSize) {
          const firstKey = this.fallbackCache.keys().next().value;
          this.fallbackCache.delete(firstKey);
        }
        this.fallbackCache.set(key, value);

        // Set TTL for fallback cache
        setTimeout(() => {
          this.fallbackCache.delete(key);
        }, ttl * 1000);
      }
      return true;
    } catch (error) {
      console.warn('Cache set error:', error.message);
      return false;
    }
  }

  async del(key) {
    try {
      if (this.isRedisAvailable) {
        await this.redis.del(key);
      } else {
        this.fallbackCache.delete(key);
      }
      return true;
    } catch (error) {
      console.warn('Cache delete error:', error.message);
      return false;
    }
  }

  async exists(key) {
    try {
      if (this.isRedisAvailable) {
        return await this.redis.exists(key);
      } else {
        return this.fallbackCache.has(key);
      }
    } catch (error) {
      console.warn('Cache exists error:', error.message);
      return this.fallbackCache.has(key);
    }
  }

  // Video-specific caching methods
  async cacheProcessedVideo(videoHash, videoBuffer, duration = 7200) {
    const key = `video:processed:${videoHash}`;
    try {
      if (this.isRedisAvailable) {
        await this.redis.setex(key, duration, videoBuffer);
      } else {
        // For large video buffers, we might want to skip fallback caching
        console.warn('Skipping video cache due to Redis unavailability');
        return false;
      }
      return true;
    } catch (error) {
      console.warn('Video cache error:', error.message);
      return false;
    }
  }

  async getCachedVideo(videoHash) {
    const key = `video:processed:${videoHash}`;
    try {
      if (this.isRedisAvailable) {
        return await this.redis.getBuffer(key);
      } else {
        return null; // Don't cache large video files in memory
      }
    } catch (error) {
      console.warn('Video cache retrieval error:', error.message);
      return null;
    }
  }

  // MIDI caching methods
  async cacheMidiData(midiHash, processedData) {
    const key = `midi:${midiHash}`;
    return await this.set(key, processedData, 3600); // 1 hour TTL
  }

  async getCachedMidiData(midiHash) {
    const key = `midi:${midiHash}`;
    return await this.get(key);
  }

  // Performance metrics caching
  async cacheMetrics(operation, duration) {
    const key = `metrics:${operation}:${Date.now()}`;
    await this.set(key, { operation, duration, timestamp: Date.now() }, 86400); // 24 hour TTL
  }

  async getMetrics(operation, since = Date.now() - 86400000) {
    try {
      if (!this.isRedisAvailable) {
        return []; // No metrics in fallback cache
      }

      const pattern = `metrics:${operation}:*`;
      const keys = await this.redis.keys(pattern);
      const metrics = [];

      for (const key of keys) {
        const data = await this.get(key);
        if (data && data.timestamp >= since) {
          metrics.push(data);
        }
      }

      return metrics;
    } catch (error) {
      console.warn('Metrics retrieval error:', error.message);
      return [];
    }
  }

  // Cache statistics and monitoring
  async getStats() {
    try {
      const stats = {
        redisAvailable: this.isRedisAvailable,
        fallbackCacheSize: this.fallbackCache.size,
        maxFallbackSize: this.maxFallbackSize,
      };

      if (this.isRedisAvailable) {
        const redisInfo = await this.redis.info('memory');
        stats.redisMemory = redisInfo;

        const keyspaceInfo = await this.redis.info('keyspace');
        stats.redisKeyspace = keyspaceInfo;
      }

      return stats;
    } catch (error) {
      return {
        redisAvailable: false,
        fallbackCacheSize: this.fallbackCache.size,
        error: error.message,
      };
    }
  }

  // Clear specific cache patterns
  async clearPattern(pattern) {
    try {
      if (this.isRedisAvailable) {
        const keys = await this.redis.keys(pattern);
        if (keys.length > 0) {
          await this.redis.del(...keys);
        }
        return keys.length;
      } else {
        // Clear fallback cache matching pattern
        let cleared = 0;
        for (const key of this.fallbackCache.keys()) {
          if (key.includes(pattern.replace('*', ''))) {
            this.fallbackCache.delete(key);
            cleared++;
          }
        }
        return cleared;
      }
    } catch (error) {
      console.warn('Cache pattern clear error:', error.message);
      return 0;
    }
  }

  async close() {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    if (this.redis) {
      await this.redis.quit();
    }

    this.fallbackCache.clear();
  }
}

export default new CacheService();
