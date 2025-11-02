/**
 * TypeScript type definitions for Claude Code hooks
 */

/**
 * Input received by hooks from Claude Code
 */
export interface HookInput {
  /** Unique session identifier */
  session_id: string;

  /** User's prompt text (for UserPromptSubmit hooks) */
  prompt?: string;

  /** Current working directory */
  working_dir?: string;

  /** Tool name that was executed (for PostToolUse hooks) */
  tool_name?: string;

  /** Tool input parameters (for PostToolUse hooks) */
  tool_input?: {
    file_path?: string;
    paths?: string[];
    [key: string]: any;
  };

  /** Tool output result (for PostToolUse hooks) */
  tool_output?: any;

  /** Transcript path for the session */
  transcript_path?: string;
}

/**
 * Skill rule configuration
 */
export interface SkillRule {
  /** Skill type (domain, guardrail, etc.) */
  type: 'domain' | 'guardrail';

  /** How to enforce the skill (suggest, warn, block) */
  enforcement: 'suggest' | 'warn' | 'block';

  /** Priority level for displaying suggestions */
  priority: 'critical' | 'high' | 'medium' | 'low';

  /** Triggers based on user prompts */
  promptTriggers?: PromptTriggers;

  /** Path patterns to match against edited files */
  pathPatterns?: string[];

  /** Content patterns to detect in files */
  contentPatterns?: string[];
}

/**
 * Prompt-based trigger configuration
 */
export interface PromptTriggers {
  /** Keywords to match in prompts (case-insensitive) */
  keywords?: string[];

  /** Regular expression patterns to detect intent */
  intentPatterns?: string[];
}

/**
 * Configuration for all skills
 */
export interface SkillRules {
  [skillName: string]: SkillRule;
}

/**
 * Matched skill with its details
 */
export interface MatchedSkill {
  /** Skill name */
  name: string;

  /** Skill rule configuration */
  rule: SkillRule;

  /** Reason why it matched */
  matchReason: 'keyword' | 'intent' | 'path' | 'content';

  /** Specific match details */
  matchDetails?: string;
}
