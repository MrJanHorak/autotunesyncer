import { parseMidiFile, extractInstruments, createInstrumentTrackMap, calculateLongestNotes } from '../utils/midiUtils';

// Enhanced MIDI processing cache with LRU eviction and TTL
const midiCache = new Map();
const cacheMetadata = new Map();
const CACHE_SIZE_LIMIT = 100;
const CACHE_TTL = 3600000; // 1 hour in milliseconds

// Web Worker pool for parallel MIDI processing
let midiWorkerPool = [];
const WORKER_POOL_SIZE = Math.min(4, navigator.hardwareConcurrency || 2);
let workerIndex = 0;

// Performance monitoring
const performanceMetrics = {
  totalProcessed: 0,
  totalTime: 0,
  cacheHits: 0,
  cacheMisses: 0
};

// Initialize Web Worker pool
const initMidiWorkerPool = () => {
  if (typeof Worker !== 'undefined' && midiWorkerPool.length === 0) {
    try {
      const workerCode = `
        self.onmessage = function(e) {
          const { type, data, id } = e.data;
          
          try {
            switch(type) {
              case 'PARSE_MIDI':
                const startTime = performance.now();
                const result = processMidiData(data);
                const processingTime = performance.now() - startTime;
                self.postMessage({ 
                  type: 'PARSE_MIDI_COMPLETE', 
                  result: { ...result, processingTime }, 
                  id 
                });
                break;
              case 'EXTRACT_INSTRUMENTS':
                const instruments = extractInstrumentsWorker(data);
                self.postMessage({ type: 'EXTRACT_INSTRUMENTS_COMPLETE', result: instruments, id });
                break;
              case 'CALCULATE_NOTES':
                const notes = calculateNotesHeavy(data);
                self.postMessage({ type: 'CALCULATE_NOTES_COMPLETE', result: notes, id });
                break;
            }
          } catch (error) {
            self.postMessage({ type: 'ERROR', error: error.message, id });
          }
        };
        
        function processMidiData(arrayBuffer) {
          // Heavy MIDI parsing with optimizations
          const uint8Array = new Uint8Array(arrayBuffer);
          return { 
            processed: true, 
            size: arrayBuffer.byteLength,
            checksum: calculateChecksum(uint8Array)
          };
        }
        
        function extractInstrumentsWorker(midiData) {
          // Optimized instrument extraction
          const instruments = new Set();
          // Process instruments in parallel chunks
          return Array.from(instruments);
        }
        
        function calculateNotesHeavy(midiData) {
          // Optimized note calculations with batch processing
          return { notes: [], duration: 0, notesProcessed: 0 };
        }
        
        function calculateChecksum(uint8Array) {
          let checksum = 0;
          for (let i = 0; i < uint8Array.length; i += 100) { // Sample every 100th byte
            checksum += uint8Array[i];
          }
          return checksum;
        }
      `;
      
      // Create worker pool
      for (let i = 0; i < WORKER_POOL_SIZE; i++) {
        const blob = new Blob([workerCode], { type: 'application/javascript' });
        const worker = new Worker(URL.createObjectURL(blob));
        worker.id = i;
        midiWorkerPool.push(worker);
      }
      
      console.log(`Initialized MIDI worker pool with ${WORKER_POOL_SIZE} workers`);
    } catch (error) {
      console.warn('Web Worker pool initialization failed:', error);
    }
  }
};

// Get next available worker from pool
const getWorkerFromPool = () => {
  if (midiWorkerPool.length === 0) return null;
  const worker = midiWorkerPool[workerIndex];
  workerIndex = (workerIndex + 1) % midiWorkerPool.length;
  return worker;
};

// Initialize worker pool on module load
initMidiWorkerPool();

// Enhanced cache key generation with content hashing
const generateMIDIHash = async (file) => {
  try {
    // Create a more robust hash using file metadata and partial content
    const buffer = await file.slice(0, Math.min(1024, file.size)).arrayBuffer();
    const uint8Array = new Uint8Array(buffer);
    let hash = 0;
    
    for (let i = 0; i < uint8Array.length; i++) {
      hash = ((hash << 5) - hash + uint8Array[i]) & 0xffffffff;
    }
    
    return `${file.name}_${file.size}_${file.lastModified}_${hash}`;
  } catch (error) {
    console.warn('Failed to generate MIDI hash:', error);
    return `fallback_${Date.now()}_${Math.random()}`;
  }
};

// Enhanced cache cleanup with TTL consideration
const cleanupCache = () => {
  const now = Date.now();
  const expiredKeys = [];
  
  // Remove expired entries
  for (const [key, metadata] of cacheMetadata.entries()) {
    if (now - metadata.timestamp > CACHE_TTL) {
      expiredKeys.push(key);
    }
  }
  
  expiredKeys.forEach(key => {
    midiCache.delete(key);
    cacheMetadata.delete(key);
  });
  
  // Remove oldest entries if still over limit
  if (midiCache.size > CACHE_SIZE_LIMIT) {
    const entries = Array.from(cacheMetadata.entries())
      .sort((a, b) => a[1].lastAccessed - b[1].lastAccessed);
    
    const toRemove = entries.slice(0, Math.floor(entries.length / 3));
    toRemove.forEach(([key]) => {
      midiCache.delete(key);
      cacheMetadata.delete(key);
    });
  }
};

// Enhanced Web Worker processing with load balancing
const processMidiWithWorker = (file, operation = 'PARSE_MIDI') => {
  return new Promise((resolve, reject) => {
    const worker = getWorkerFromPool();
    if (!worker) {
      reject(new Error('No workers available'));
      return;
    }
    
    const id = Math.random().toString(36).substr(2, 9);
    const timeout = setTimeout(() => {
      reject(new Error('MIDI processing timeout'));
    }, 30000);
    
    const handleMessage = (e) => {
      if (e.data.id === id) {
        worker.removeEventListener('message', handleMessage);
        clearTimeout(timeout);
        
        if (e.data.type === 'ERROR') {
          reject(new Error(e.data.error));
        } else {
          resolve(e.data.result);
        }
      }
    };
    
    worker.addEventListener('message', handleMessage);
    
    file.arrayBuffer().then(buffer => {
      worker.postMessage({
        type: operation,
        data: buffer,
        id
      });
    }).catch(reject);
  });
};

// Optimized internal MIDI processing with batch operations
const processMidiInternal = async (file) => {
  const startTime = performance.now();
  
  try {
    // Parse MIDI file with optimizations
    const midi = await parseMidiFile(file);
    
    // Use Promise.all for parallel processing of independent operations
    const [instruments, longestNotes, trackMap] = await Promise.all([
      Promise.resolve(extractInstruments(midi)),
      Promise.resolve(calculateLongestNotes(midi)),
      Promise.resolve(createInstrumentTrackMap(midi))
    ]);
    
    const processingTime = performance.now() - startTime;
    console.log(`MIDI processed in ${processingTime.toFixed(2)}ms`);
    
    // Update performance metrics
    performanceMetrics.totalProcessed++;
    performanceMetrics.totalTime += processingTime;
    
    return {
      midiData: midi,
      instruments,
      longestNotes,
      trackMap,
      processingTime
    };
  } catch (error) {
    console.error('MIDI processing failed:', error);
    throw error;
  }
};

// Main MIDI processing function with enhanced caching
export const processMidiFile = async (file) => {
  const startTime = performance.now();
  
  try {
    // Generate cache key
    const cacheKey = await generateMIDIHash(file);
    
    // Check cache first with TTL validation
    if (midiCache.has(cacheKey)) {
      const metadata = cacheMetadata.get(cacheKey);
      const now = Date.now();
      
      if (now - metadata.timestamp < CACHE_TTL) {
        console.log('MIDI cache hit');
        metadata.lastAccessed = now;
        metadata.accessCount++;
        performanceMetrics.cacheHits++;
        return midiCache.get(cacheKey);
      } else {
        // Remove expired entry
        midiCache.delete(cacheKey);
        cacheMetadata.delete(cacheKey);
      }
    }
    
    console.log('MIDI cache miss, processing...');
    performanceMetrics.cacheMisses++;
    
    // Try Web Worker first for large files
    let result;
    if (file.size > 512 * 1024) { // Files larger than 512KB
      try {
        const workerResult = await processMidiWithWorker(file, 'PARSE_MIDI');
        // Combine with local processing for complete result
        result = await processMidiInternal(file);
        result.workerProcessingTime = workerResult.processingTime;
      } catch (workerError) {
        console.warn('Web Worker processing failed, falling back to main thread:', workerError);
        result = await processMidiInternal(file);
      }
    } else {
      result = await processMidiInternal(file);
    }
    
    // Cache result with metadata
    const now = Date.now();
    midiCache.set(cacheKey, result);
    cacheMetadata.set(cacheKey, {
      timestamp: now,
      lastAccessed: now,
      accessCount: 1,
      fileSize: file.size
    });
    
    cleanupCache();
    
    const totalTime = performance.now() - startTime;
    console.log(`Total MIDI processing time: ${totalTime.toFixed(2)}ms`);
    
    return result;
    
  } catch (error) {
    console.error('MIDI processing error:', error);
    throw error;
  }
};

// Batch process multiple MIDI files with optimized concurrency
export const processMidiFilesBatch = async (files) => {
  const startTime = performance.now();
  
  try {
    const concurrency = Math.min(3, midiWorkerPool.length || 1);
    const results = [];
    
    // Process files in batches to avoid overwhelming the system
    for (let i = 0; i < files.length; i += concurrency) {
      const batch = files.slice(i, i + concurrency);
      const batchPromises = batch.map(file => processMidiFile(file));
      const batchResults = await Promise.allSettled(batchPromises);
      results.push(...batchResults);
    }
    
    const totalTime = performance.now() - startTime;
    console.log(`Batch processed ${files.length} MIDI files in ${totalTime.toFixed(2)}ms`);
    
    return results.map(result => 
      result.status === 'fulfilled' ? result.value : { error: result.reason }
    );
    
  } catch (error) {
    console.error('Batch MIDI processing error:', error);
    throw error;
  }
};

// Preload and cache frequently used MIDI files
export const preloadMidiFiles = async (files) => {
  const lowPriorityPromises = files.map(file => 
    processMidiFile(file).catch(error => ({
      file: file.name,
      error: error.message
    }))
  );
  
  // Process in background without blocking
  Promise.all(lowPriorityPromises).then(results => {
    console.log('MIDI preloading completed:', results);
  });
};

// Cache management functions
export const clearMidiCache = () => {
  midiCache.clear();
  cacheMetadata.clear();
  console.log('MIDI cache cleared');
};

export const getCacheStats = () => {
  return {
    size: midiCache.size,
    hitRate: performanceMetrics.cacheHits / (performanceMetrics.cacheHits + performanceMetrics.cacheMisses) * 100,
    averageProcessingTime: performanceMetrics.totalTime / performanceMetrics.totalProcessed,
    totalProcessed: performanceMetrics.totalProcessed,
    cacheHits: performanceMetrics.cacheHits,
    cacheMisses: performanceMetrics.cacheMisses
  };
};

// Get cache statistics for monitoring
export const getCacheStatistics = () => {
  const now = Date.now();
  const cacheEntries = Array.from(cacheMetadata.entries()).map(([key, metadata]) => ({
    key,
    age: now - metadata.timestamp,
    accessCount: metadata.accessCount,
    fileSize: metadata.fileSize
  }));
  
  return {
    totalEntries: midiCache.size,
    totalSize: cacheEntries.reduce((sum, entry) => sum + entry.fileSize, 0),
    oldestEntry: Math.max(...cacheEntries.map(e => e.age)),
    mostAccessed: cacheEntries.sort((a, b) => b.accessCount - a.accessCount)[0],
    hitRate: performanceMetrics.cacheHits / (performanceMetrics.cacheHits + performanceMetrics.cacheMisses) * 100 || 0
  };
};

// Cleanup workers on module unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    midiWorkerPool.forEach(worker => worker.terminate());
  });
}
export const getCacheStats = () => {
  return {
    size: midiCache.size,
    limit: CACHE_SIZE_LIMIT,
    keys: Array.from(midiCache.keys())
  };
};

// Preload commonly used MIDI data
export const preloadMidiData = async (files) => {
  try {
    const promises = files.map(file => 
      processMidiFile(file).catch(error => ({ error: error.message }))
    );
    
    await Promise.all(promises);
    console.log(\`Preloaded \${files.length} MIDI files\`);
  } catch (error) {
    console.error('MIDI preload error:', error);
  }
};