import { appendFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import process from 'process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const LOG_DIR = join(__dirname, '../logs');

// Ensure log directory exists
if (!existsSync(LOG_DIR)) {
  mkdirSync(LOG_DIR, { recursive: true });
}

class FileLogger {
  constructor(filename = 'app.log') {
    this.logPath = join(LOG_DIR, filename);
  }

  _writeLog(level, message, meta = {}) {
    const timestamp = new Date().toISOString();
    const logEntry = JSON.stringify({
      timestamp,
      level,
      message,
      ...meta
    }) + '\n';

    // Write to file
    appendFileSync(this.logPath, logEntry);
    
    // Also log to console in development
    if (process.env.NODE_ENV !== 'production') {
      console.log(`${timestamp} [${level}]: ${message}`, meta);
    }
  }

  info(message, meta) { this._writeLog('info', message, meta); }
  error(message, meta) { this._writeLog('error', message, meta); }
  debug(message, meta) { this._writeLog('debug', message, meta); }
}

export default new FileLogger('composition.log');