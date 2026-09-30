/**
 * ALIZIA AI - Account, Organization & API Key Service
 * Implements PRD Sections 11, 12, 13, 111.
 */

import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import crypto from 'crypto';

const PORT = parseInt(process.env.PORT || '3001', 10);

const server: FastifyInstance = Fastify({
  logger: true,
});

server.register(cors, { origin: '*' });

// In-memory data store for MVP accounts & keys
interface StoredAPIKey {
  id: string;
  organizationId: string;
  name: string;
  keyPrefix: string;
  keyHash: string;
  scopes: string[];
  createdAt: number;
  revokedAt?: number;
}

const apiKeysDb: Map<string, StoredAPIKey> = new Map();

/**
 * Section 13: Generate cryptographically secure API key
 * Format: alz_live_xxxxxxxxxx or alz_test_xxxxxxxxxx
 * Store only hashed credentials.
 */
function generateApiKey(env: 'live' | 'test' = 'live'): { key: string; keyPrefix: string; keyHash: string } {
  const prefix = `alz_${env}_`;
  const secretPart = crypto.randomBytes(24).toString('hex');
  const fullKey = `${prefix}${secretPart}`;
  const keyHash = crypto.createHash('sha256').update(fullKey).digest('hex');

  return { key: fullKey, keyPrefix: prefix, keyHash };
}

server.get('/health', async () => ({
  status: 'healthy',
  service: 'alizia-account-service',
  timestamp: Date.now(),
}));

// Create API Key
server.post('/v1/accounts/api-keys', async (req: FastifyRequest, reply: FastifyReply) => {
  const body = req.body as any || {};
  const orgId = body.organizationId || 'org_default';
  const name = body.name || 'Default Production Key';
  const env = body.env === 'test' ? 'test' : 'live';
  const scopes = body.scopes || ['files.read', 'code.execute', 'internet.search'];

  const { key, keyPrefix, keyHash } = generateApiKey(env);
  const keyId = `key_${crypto.randomBytes(8).toString('hex')}`;

  const record: StoredAPIKey = {
    id: keyId,
    organizationId: orgId,
    name,
    keyPrefix,
    keyHash,
    scopes,
    createdAt: Date.now(),
  };

  apiKeysDb.set(keyHash, record);

  // Return the full secret key ONCE. In future responses, only prefix is shown.
  return reply.status(201).send({
    id: keyId,
    name,
    key, // Only returned on creation
    keyPrefix,
    scopes,
    createdAt: record.createdAt,
  });
});

// Verify API Key (Used by Gateway / Auth middleware)
server.post('/v1/accounts/api-keys/verify', async (req: FastifyRequest, reply: FastifyReply) => {
  const body = req.body as any || {};
  const token = body.apiKey || '';

  if (!token.startsWith('alz_live_') && !token.startsWith('alz_test_')) {
    return reply.status(401).send({ valid: false, reason: 'Invalid key prefix format' });
  }

  const hash = crypto.createHash('sha256').update(token).digest('hex');
  const found = apiKeysDb.get(hash);

  if (!found || found.revokedAt) {
    return reply.status(401).send({ valid: false, reason: 'API key not found or revoked' });
  }

  return reply.send({
    valid: true,
    keyId: found.id,
    organizationId: found.organizationId,
    scopes: found.scopes,
  });
});

export async function startServer() {
  try {
    await server.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`[Account Service] Listening on http://0.0.0.0:${PORT}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

if (require.main === module) {
  startServer();
}

export { server };
