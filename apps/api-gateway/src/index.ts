/**
 * ALIZIA AI - Global API Gateway (TypeScript / Fastify)
 * Implements PRD Sections 1, 7, 8, 13, 161, 162, 176.
 */

import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import crypto from 'crypto';

const PORT = parseInt(process.env.PORT || '3000', 10);
const AI_ORCHESTRATOR_URL = process.env.AI_ORCHESTRATOR_URL || 'http://127.0.0.1:8000';

const server: FastifyInstance = Fastify({
  logger: {
    level: process.env.LOG_LEVEL || 'info',
  },
  requestIdHeader: 'x-request-id',
  genReqId: () => `req_${crypto.randomBytes(8).toString('hex')}`,
});

// Register CORS
server.register(cors, {
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
});

// Middleware: Request ID & Rate Limiting headers (Sections 162, 176)
server.addHook('onRequest', async (req: FastifyRequest, reply: FastifyReply) => {
  reply.header('x-request-id', req.id);
  reply.header('x-ratelimit-limit-requests', '10000');
  reply.header('x-ratelimit-remaining-requests', '9985');
  reply.header('x-ratelimit-reset-requests', '60');
  reply.header('x-ratelimit-limit-tokens', '2000000');
});

// Health check endpoint
server.get('/health', async () => {
  return {
    status: 'healthy',
    gateway: 'alizia-global-api-gateway',
    uptime: process.uptime(),
    timestamp: Date.now(),
    downstream: AI_ORCHESTRATOR_URL,
  };
});

// Gateway Forwarding / Reverse Proxy to AI Orchestrator (POST /v1/responses)
server.post('/v1/responses', async (req: FastifyRequest, reply: FastifyReply) => {
  const reqBody = req.body as any;
  const isStreaming = reqBody && reqBody.stream === true;

  try {
    const upstreamRes = await fetch(`${AI_ORCHESTRATOR_URL}/v1/responses`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-request-id': req.id,
        ...(req.headers.authorization ? { authorization: req.headers.authorization } : {}),
      },
      body: JSON.stringify(reqBody),
    });

    if (isStreaming && upstreamRes.body) {
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'x-request-id': req.id,
      });

      const reader = upstreamRes.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        reply.raw.write(Buffer.from(value));
      }
      reply.raw.end();
      return;
    }

    const data = await upstreamRes.json();
    return reply.status(upstreamRes.status).send(data);
  } catch (err: any) {
    req.log.error(err, 'Failed to proxy response to AI orchestrator');
    return reply.status(502).send({
      error: {
        type: 'api_gateway_error',
        code: 'upstream_unavailable',
        message: 'Could not connect to Alizia AI Orchestrator service.',
        request_id: req.id,
      },
    });
  }
});

// Proxy GET /v1/models
server.get('/v1/models', async (req: FastifyRequest, reply: FastifyReply) => {
  try {
    const upstreamRes = await fetch(`${AI_ORCHESTRATOR_URL}/v1/models`);
    const data = await upstreamRes.json();
    return reply.status(upstreamRes.status).send(data);
  } catch (err) {
    return reply.status(502).send({
      error: {
        type: 'api_gateway_error',
        code: 'upstream_unavailable',
        message: 'Could not connect to Alizia AI Orchestrator service.',
        request_id: req.id,
      },
    });
  }
});

// Proxy POST /v1/embeddings
server.post('/v1/embeddings', async (req: FastifyRequest, reply: FastifyReply) => {
  try {
    const upstreamRes = await fetch(`${AI_ORCHESTRATOR_URL}/v1/embeddings`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-request-id': req.id,
      },
      body: JSON.stringify(req.body),
    });
    const data = await upstreamRes.json();
    return reply.status(upstreamRes.status).send(data);
  } catch (err) {
    return reply.status(502).send({
      error: {
        type: 'api_gateway_error',
        code: 'upstream_unavailable',
        message: 'Could not connect to Alizia AI Orchestrator service.',
        request_id: req.id,
      },
    });
  }
});

// Start Gateway
export async function startServer() {
  try {
    await server.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`[Alizia Gateway] Listening on http://0.0.0.0:${PORT}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

if (require.main === module) {
  startServer();
}

export { server };
