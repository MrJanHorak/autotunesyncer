import logger from '../utils/logger.js';

export const requestLogger = (req, res, next) => {
  const start = Date.now();
  const requestId = req.headers['x-request-id'] || crypto.randomUUID();
  
  logger.info('Request started', {
    requestId,
    method: req.method,
    url: req.url,
    body: req.body,
    headers: req.headers
  });

  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info('Request completed', {
      requestId,
      duration,
      statusCode: res.statusCode
    });
  });

  next();
};