/**
 * ALIZIA AI - Billing, Usage Ledger & Quota Service
 * Implements PRD Sections 104, 105, 106, 107, 110.
 */

import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import crypto from 'crypto';

const PORT = parseInt(process.env.PORT || '3002', 10);

const server: FastifyInstance = Fastify({
  logger: true,
});

server.register(cors, { origin: '*' });

// Section 106: Immutable Usage Ledger Data
interface UsageRecord {
  eventId: string;
  organizationId: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cost: number;
  idempotencyKey?: string;
  timestamp: number;
}

const ledger: UsageRecord[] = [];
const processedIdempotencyKeys: Set<string> = new Set();

// Section 105: Pricing Configuration (per 1,000,000 tokens)
const MODEL_PRICING: Record<string, { input: number; output: number; reasoning: number }> = {
  'alizia-nova': { input: 3.00, output: 15.00, reasoning: 15.00 },
  'alizia-pulse': { input: 0.15, output: 0.60, reasoning: 0.60 },
  'alizia-forge': { input: 2.00, output: 8.00, reasoning: 8.00 },
  'alizia-vision': { input: 2.50, output: 10.00, reasoning: 10.00 },
  'alizia-embed-v1': { input: 0.05, output: 0.00, reasoning: 0.00 },
};

server.get('/health', async () => ({
  status: 'healthy',
  service: 'alizia-billing-service',
  timestamp: Date.now(),
}));

// Record Usage Event (Section 106)
server.post('/v1/billing/usage', async (req: FastifyRequest, reply: FastifyReply) => {
  const body = req.body as any || {};
  const idempotencyKey = req.headers['idempotency-key'] as string | undefined;

  // Section 110: Idempotency enforcement
  if (idempotencyKey) {
    if (processedIdempotencyKeys.has(idempotencyKey)) {
      return reply.status(200).send({
        status: 'idempotent_replay',
        message: 'Usage event previously recorded',
      });
    }
    processedIdempotencyKeys.add(idempotencyKey);
  }

  const model = body.model || 'alizia-nova';
  const inputTokens = body.input_tokens || 0;
  const outputTokens = body.output_tokens || 0;
  const reasoningTokens = body.reasoning_tokens || 0;

  const rates = MODEL_PRICING[model] || MODEL_PRICING['alizia-nova'];
  const cost =
    (inputTokens / 1_000_000) * rates.input +
    (outputTokens / 1_000_000) * rates.output +
    (reasoningTokens / 1_000_000) * rates.reasoning;

  const record: UsageRecord = {
    eventId: `evt_${crypto.randomBytes(8).toString('hex')}`,
    organizationId: body.organization_id || 'org_default',
    model,
    inputTokens,
    outputTokens,
    reasoningTokens,
    cost: parseFloat(cost.toFixed(6)),
    idempotencyKey,
    timestamp: Date.now(),
  };

  ledger.push(record);

  return reply.status(201).send(record);
});

// Organization Usage Summary & Quota Status (Section 107)
server.get('/v1/billing/organizations/:orgId/usage', async (req: FastifyRequest, reply: FastifyReply) => {
  const { orgId } = req.params as { orgId: string };
  const records = ledger.filter((r) => r.organizationId === orgId);

  const totalInputTokens = records.reduce((acc, r) => acc + r.inputTokens, 0);
  const totalOutputTokens = records.reduce((acc, r) => acc + r.outputTokens, 0);
  const totalCost = records.reduce((acc, r) => acc + r.cost, 0);

  const monthlyQuotaTokens = 100_000_000;

  return reply.send({
    organizationId: orgId,
    usage: {
      totalInputTokens,
      totalOutputTokens,
      totalTokens: totalInputTokens + totalOutputTokens,
      totalSpendUSD: parseFloat(totalCost.toFixed(4)),
    },
    quota: {
      monthlyTokenQuota: monthlyQuotaTokens,
      consumedPercent: parseFloat(((totalInputTokens + totalOutputTokens) / monthlyQuotaTokens * 100).toFixed(2)),
      remainingTokens: Math.max(0, monthlyQuotaTokens - (totalInputTokens + totalOutputTokens)),
    },
  });
});

export async function startServer() {
  try {
    await server.listen({ port: PORT, host: '0.0.0.0' });
    console.log(`[Billing Service] Listening on http://0.0.0.0:${PORT}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
}

if (require.main === module) {
  startServer();
}

export { server };
