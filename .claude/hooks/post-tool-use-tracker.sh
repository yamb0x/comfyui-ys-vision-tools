#!/bin/bash
set -e

# Post-Tool-Use Tracker Hook
# Bash wrapper for TypeScript implementation

cd "$CLAUDE_PROJECT_DIR/.claude/hooks"
cat | npx tsx post-tool-use-tracker.ts
