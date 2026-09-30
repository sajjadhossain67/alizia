-- ============================================================================
-- ALIZIA AI - Production Database Schema (PostgreSQL + pgvector)
-- In accordance with Sections 10, 78, 111-115, 149 of the PRD
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
-- Enable pgvector if installed
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS "vector";
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension not installed or permission denied. Falling back to bytea/float arrays if needed.';
END $$;

-- ----------------------------------------------------------------------------
-- 1. Organizations & Tenants (Section 78 Data Isolation)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    tier VARCHAR(50) DEFAULT 'pro', -- free, go, pro, business, enterprise
    settings JSONB DEFAULT '{}'::jsonb, -- enterprise control plane (Section 123)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    avatar_url TEXT,
    system_role VARCHAR(50) DEFAULT 'user', -- user, superadmin
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS organization_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'member', -- owner, admin, developer, member, viewer, billing_admin, security_admin
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, user_id)
);

CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 2. API Keys (Section 13: alz_live_*, alz_test_*, hashed credentials)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(32) NOT NULL, -- e.g. alz_live_ or alz_test_
    key_hash VARCHAR(128) NOT NULL UNIQUE, -- SHA-256 hash of the full secret key
    scopes TEXT[] DEFAULT ARRAY['files.read', 'code.execute', 'internet.search'],
    ip_restrictions CIDR[],
    rate_limit_rpm INT DEFAULT 600,
    rate_limit_tpm INT DEFAULT 200000,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMPTZ
);

-- ----------------------------------------------------------------------------
-- 3. Conversations & Messages (Section 112, 113, Tree-based branching)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    model_id VARCHAR(100) NOT NULL DEFAULT 'alizia-nova',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_message_id UUID REFERENCES messages(id) ON DELETE SET NULL, -- Branching support (Section 25)
    role VARCHAR(32) NOT NULL, -- system, developer, user, assistant, tool
    content JSONB NOT NULL, -- Content array or text per Section 26
    reasoning_summary TEXT, -- Section 7 & 15: Exposed reasoning summary
    token_count INT DEFAULT 0,
    model_version VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 4. Files, Intelligence & Vector Embeddings (Sections 21, 28, 31-35, 111)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    filename VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    byte_size BIGINT NOT NULL,
    s3_key TEXT NOT NULL,
    parsed_metadata JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(50) DEFAULT 'ready', -- pending, scanning, parsing, ready, failed
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    file_id UUID REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding JSONB, -- Stored as JSON array or pgvector
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope VARCHAR(50) NOT NULL, -- session, user, workspace, agent (Section 27)
    content TEXT NOT NULL,
    embedding JSONB,
    importance REAL DEFAULT 0.5, -- 0.0 - 1.0 (Section 28)
    confidence REAL DEFAULT 0.9, -- 0.0 - 1.0 (Section 28)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 5. Agents, Runs, Steps & Tool Executions (Sections 39-44, 48-49, 114, 115)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued', -- queued, planning, running, waiting_for_tool, waiting_for_user, verifying, completed, failed, cancelled, expired
    current_step INT DEFAULT 0,
    max_steps INT DEFAULT 100,
    token_budget BIGINT DEFAULT 1000000,
    cost_budget DECIMAL(12, 4) DEFAULT 10.0000,
    tokens_consumed BIGINT DEFAULT 0,
    cost_consumed DECIMAL(12, 4) DEFAULT 0.0000,
    plan JSONB DEFAULT '[]'::jsonb, -- Section 42
    evidence JSONB DEFAULT '[]'::jsonb, -- Section 187, 191 Verifiable AI
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INT NOT NULL,
    action_type VARCHAR(64) NOT NULL, -- plan, tool_call, reasoning, verifier
    input_payload JSONB DEFAULT '{}'::jsonb,
    output_payload JSONB DEFAULT '{}'::jsonb,
    status VARCHAR(32) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_run_id UUID REFERENCES agent_runs(id) ON DELETE CASCADE,
    tool_name VARCHAR(128) NOT NULL,
    arguments JSONB NOT NULL,
    permission_scope VARCHAR(128) NOT NULL,
    risk_level INT NOT NULL DEFAULT 0, -- R0 to R4 (Section 74)
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, waiting_confirmation, running, success, failed, denied
    confirmation_token VARCHAR(255), -- Section 75
    execution_result JSONB,
    execution_id VARCHAR(128) UNIQUE NOT NULL, -- Cryptographically unique execution ID (Section 49)
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_run_id UUID REFERENCES agent_runs(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(64) NOT NULL, -- code_patch, document, report, spreadsheet, dataset (Section 148)
    content TEXT NOT NULL,
    s3_url TEXT,
    version INT DEFAULT 1,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 6. Billing, Usage Ledger & Quotas (Sections 104-107)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    model VARCHAR(100) NOT NULL,
    input_tokens INT DEFAULT 0,
    cached_input_tokens INT DEFAULT 0,
    output_tokens INT DEFAULT 0,
    reasoning_tokens INT DEFAULT 0,
    cost DECIMAL(12, 6) DEFAULT 0.000000,
    idempotency_key VARCHAR(255) UNIQUE, -- Section 110
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 7. Audit Logging & Webhooks (Sections 70, 116, 177)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL, -- login, logout, api_key_created, tool_executed, etc.
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- Append-only
);

CREATE TABLE IF NOT EXISTS webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    secret VARCHAR(255) NOT NULL, -- HMAC signature secret
    subscribed_events TEXT[] NOT NULL, -- response.completed, agent.step.completed, etc.
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- Indexes for High Throughput & Isolation (Section 78)
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_conv_org_user ON conversations(organization_id, user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_memories_org_user ON memories(organization_id, user_id, scope);
CREATE INDEX IF NOT EXISTS idx_agent_runs_org ON agent_runs(organization_id, status);
CREATE INDEX IF NOT EXISTS idx_tool_exec_run ON tool_executions(agent_run_id);
CREATE INDEX IF NOT EXISTS idx_usage_org_created ON usage_events(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_org_created ON audit_events(organization_id, created_at DESC);
