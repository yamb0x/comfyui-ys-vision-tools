#!/usr/bin/env node

/**
 * Skill Activation Hook (UserPromptSubmit)
 *
 * This hook runs BEFORE Claude sees the user's prompt and suggests
 * relevant skills based on keyword and intent pattern matching.
 *
 * Flow:
 * 1. Read JSON input from stdin (user prompt + context)
 * 2. Load skill-rules.json configuration
 * 3. Match prompt against keywords and intent patterns
 * 4. Output formatted skill suggestions to stdout
 * 5. Claude receives suggestions as additional context
 */

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import type { HookInput, SkillRules, MatchedSkill } from './types/hook-types.js';

// Get directory paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CLAUDE_DIR = join(__dirname, '..');
const SKILLS_DIR = join(CLAUDE_DIR, 'skills');
const SKILL_RULES_PATH = join(SKILLS_DIR, 'skill-rules.json');

/**
 * Read and parse stdin input
 */
async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');

    process.stdin.on('data', (chunk) => {
      data += chunk;
    });

    process.stdin.on('end', () => {
      resolve(data);
    });

    process.stdin.on('error', (error) => {
      reject(error);
    });
  });
}

/**
 * Load skill rules configuration
 */
function loadSkillRules(): SkillRules {
  try {
    const content = readFileSync(SKILL_RULES_PATH, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error('Error loading skill-rules.json:', error);
    return {};
  }
}

/**
 * Check if prompt matches skill keywords
 */
function matchesKeywords(prompt: string, keywords: string[] = []): boolean {
  const lowerPrompt = prompt.toLowerCase();
  return keywords.some(keyword => lowerPrompt.includes(keyword.toLowerCase()));
}

/**
 * Check if prompt matches intent patterns
 */
function matchesIntentPattern(prompt: string, patterns: string[] = []): boolean {
  const lowerPrompt = prompt.toLowerCase();
  return patterns.some(pattern => {
    try {
      const regex = new RegExp(pattern, 'i');
      return regex.test(lowerPrompt);
    } catch (error) {
      console.error(`Invalid regex pattern: ${pattern}`, error);
      return false;
    }
  });
}

/**
 * Find skills that match the user's prompt
 */
function findMatchingSkills(prompt: string, skillRules: SkillRules): MatchedSkill[] {
  const matches: MatchedSkill[] = [];

  for (const [skillName, rule] of Object.entries(skillRules)) {
    if (!rule.promptTriggers) continue;

    const { keywords = [], intentPatterns = [] } = rule.promptTriggers;

    // Check keyword matches
    if (matchesKeywords(prompt, keywords)) {
      matches.push({
        name: skillName,
        rule,
        matchReason: 'keyword',
        matchDetails: keywords.find(k => prompt.toLowerCase().includes(k.toLowerCase()))
      });
      continue;
    }

    // Check intent pattern matches
    if (matchesIntentPattern(prompt, intentPatterns)) {
      matches.push({
        name: skillName,
        rule,
        matchReason: 'intent',
        matchDetails: intentPatterns.find(p => {
          try {
            return new RegExp(p, 'i').test(prompt.toLowerCase());
          } catch {
            return false;
          }
        })
      });
    }
  }

  return matches;
}

/**
 * Format skill suggestions for Claude
 */
function formatSuggestions(matches: MatchedSkill[]): string {
  if (matches.length === 0) {
    return '';
  }

  // Group by priority
  const byPriority = {
    critical: matches.filter(m => m.rule.priority === 'critical'),
    high: matches.filter(m => m.rule.priority === 'high'),
    medium: matches.filter(m => m.rule.priority === 'medium'),
    low: matches.filter(m => m.rule.priority === 'low')
  };

  let output = '\n════════════════════════════════════════════════════\n';
  output += '📚 RELEVANT SKILLS DETECTED\n';
  output += '════════════════════════════════════════════════════\n\n';

  if (byPriority.critical.length > 0) {
    output += '⚠️  CRITICAL SKILLS (REQUIRED):\n';
    byPriority.critical.forEach(m => {
      output += `   • ${m.name}\n`;
    });
    output += '\n';
  }

  if (byPriority.high.length > 0) {
    output += '📚 RECOMMENDED SKILLS:\n';
    byPriority.high.forEach(m => {
      output += `   • ${m.name}\n`;
    });
    output += '\n';
  }

  if (byPriority.medium.length > 0) {
    output += '💡 SUGGESTED SKILLS:\n';
    byPriority.medium.forEach(m => {
      output += `   • ${m.name}\n`;
    });
    output += '\n';
  }

  if (byPriority.low.length > 0) {
    output += '📌 OPTIONAL SKILLS:\n';
    byPriority.low.forEach(m => {
      output += `   • ${m.name}\n`;
    });
    output += '\n';
  }

  output += 'Use the Skill tool to load these skills if relevant to your task.\n';
  output += '════════════════════════════════════════════════════\n';

  return output;
}

/**
 * Main execution
 */
async function main() {
  try {
    // Read input from stdin
    const input = await readStdin();
    const hookInput: HookInput = JSON.parse(input);

    // Validate input
    if (!hookInput.prompt) {
      console.error('No prompt provided in hook input');
      process.exit(0);
    }

    // Load skill rules
    const skillRules = loadSkillRules();

    // Find matching skills
    const matches = findMatchingSkills(hookInput.prompt, skillRules);

    // Output formatted suggestions
    if (matches.length > 0) {
      const suggestions = formatSuggestions(matches);
      console.log(suggestions);
    }

    process.exit(0);
  } catch (error) {
    console.error('Error in skill-activation-prompt hook:', error);
    process.exit(1);
  }
}

// Run main
main();
