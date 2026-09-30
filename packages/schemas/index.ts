/**
 * ALIZIA AI - Core TypeScript Protocol Definitions
 * Implements PRD Sections 14-16, 26, 40-49, 74-76, 111-115, 161
 */

export enum ReasoningEffort {
  NONE = 'none',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  EXTREME = 'extreme',
  AUTO = 'auto',
}

export enum MessageRole {
  SYSTEM = 'system',
  DEVELOPER = 'developer',
  USER = 'user',
  ASSISTANT = 'assistant',
  TOOL = 'tool',
}

export enum TrustLevel {
  TRUSTED = 'trusted',
  AUTHORIZED = 'authorized',
  UNTRUSTED_DATA = 'untrusted_data',
}

export enum RiskLevel {
  R0_INFORMATIONAL = 0,
  R1_REVERSIBLE = 1,
  R2_EXTERNALLY_VISIBLE = 2,
  R3_SENSITIVE = 3,
  R4_DESTRUCTIVE = 4,
}

export enum AgentState {
  QUEUED = 'queued',
  PLANNING = 'planning',
  RUNNING = 'running',
  WAITING_FOR_TOOL = 'waiting_for_tool',
  WAITING_FOR_USER = 'waiting_for_user',
  VERIFYING = 'verifying',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  EXPIRED = 'expired',
}

export interface ContentPart {
  type: string;
  text?: string;
  media_url?: string;
  media_mime_type?: string;
  trust_level?: TrustLevel;
}

export interface MessageItem {
  id?: string;
  role: MessageRole;
  content: string | ContentPart[];
  parent_message_id?: string;
  created_at?: number;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
  permission_scope?: string;
  risk_level?: RiskLevel;
}

export interface CreateResponseRequest {
  model?: string;
  input: MessageItem[];
  reasoning?: {
    effort: ReasoningEffort;
  };
  tools?: ToolDefinition[];
  response_format?: {
    type: 'text' | 'json_object' | 'json_schema';
    schema?: Record<string, any>;
  };
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  workspace_id?: string;
  conversation_id?: string;
}

export interface TokenUsage {
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
}

export interface ResponseObject {
  id: string;
  object: 'response';
  created_at: number;
  model: string;
  status: 'completed' | 'in_progress' | 'failed' | 'requires_action';
  output: any[];
  reasoning_summary?: string;
  usage: TokenUsage;
  citations?: any[];
}

export interface APIErrorResponse {
  error: {
    type: string;
    code: string;
    message: string;
    request_id: string;
  };
}
