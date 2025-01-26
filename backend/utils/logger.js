import winston from 'winston';
import { join } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { existsSync, mkdirSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const logDir = join(__dirname, '../logs');

if (!existsSync(logDir)) {
  mkdirSync(logDir, { recursive: true });
}

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json(),
    winston.format.printf(({ timestamp, level, message, ...rest }) => {
      return JSON.stringify({
        timestamp,
        level,
        message,
        ...rest,
      });
    })
  ),
  transports: [
    new winston.transports.File({
      filename: join(logDir, 'composition-error.log'),
      level: 'error',
      handleExceptions: true,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
    }),
    new winston.transports.File({
      filename: join(logDir, 'composition.log'),
      level: 'info',
      handleExceptions: true,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
      options: { flags: 'a' },
    }),
    new winston.transports.Console({
      format: winston.format.simple(),
      handleExceptions: true,
    }),
  ],
  exitOnError: false
});

// Add explicit flush method
logger.flush = () => {
  logger.transports.forEach((transport) => {
    if (typeof transport.flush === 'function') {
      transport.flush();
    }
  });
};

export default logger;