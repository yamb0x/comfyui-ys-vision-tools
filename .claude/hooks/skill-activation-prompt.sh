#!/bin/bash
set -e

# Skill Activation Hook (UserPromptSubmit)
# Bash wrapper for TypeScript implementation

cd "$CLAUDE_PROJECT_DIR/.claude/hooks"
cat | npx tsx skill-activation-prompt.ts
