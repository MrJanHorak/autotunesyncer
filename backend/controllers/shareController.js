import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { v4 as uuidv4 } from 'uuid';

const BUCKET = process.env.S3_BUCKET;
const REGION = process.env.AWS_REGION || 'us-east-1';
const SHARE_URL_EXPIRY_SECONDS = parseInt(process.env.SHARE_URL_EXPIRY || '604800', 10); // 7 days

let _s3 = null;
function getS3() {
  if (!_s3) {
    _s3 = new S3Client({ region: REGION });
  }
  return _s3;
}

export const shareComposition = async (req, res) => {
  if (!BUCKET) {
    return res.status(501).json({
      error: 'Share links are not configured on this server (S3_BUCKET env var not set)',
    });
  }

  if (!req.file) {
    return res.status(400).json({ error: 'Video file required' });
  }

  const key = `compositions/${req.user.id}/${uuidv4()}.mp4`;

  try {
    await getS3().send(
      new PutObjectCommand({
        Bucket: BUCKET,
        Key: key,
        Body: req.file.buffer,
        ContentType: 'video/mp4',
      }),
    );

    const url = await getSignedUrl(
      getS3(),
      new GetObjectCommand({ Bucket: BUCKET, Key: key }),
      { expiresIn: SHARE_URL_EXPIRY_SECONDS },
    );

    res.json({ url, expiresInSeconds: SHARE_URL_EXPIRY_SECONDS });
  } catch (err) {
    console.error('[share] S3 upload error:', err);
    res.status(500).json({ error: 'Upload to cloud storage failed' });
  }
};
