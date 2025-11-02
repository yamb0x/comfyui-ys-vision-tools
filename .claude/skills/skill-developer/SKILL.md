---
name: skill-developer
description: Create and manage Claude Code skills following Anthropic best practices. Use when creating new skills, modifying skill-rules.json, understanding trigger patterns, working with hooks, debugging skill activation, or implementing progressive disclosure. Covers skill structure, YAML frontmatter, trigger types (keywords, intent patterns, file paths, content patterns), enforcement levels (block, suggest, warn), hook mechanisms (UserPromptSubmit, PostToolUse), session tracking, and the 500-line rule.
---

# Skill Developer Guide

## Purpose

Comprehensive guide for creating and managing skills in Claude Code with auto-activation system, following Anthropic's official best practices including the 500-line rule and progressive disclosure pattern.

---

## When to Use This Skill

Automatically activates when you mention:

- Creating or adding skills
- Modifying skill triggers or rules
- Understanding how skill activation works
- Debugging skill activation issues
- Working with skill-rules.json
- Hook system mechanics
- Claude Code best practices
- Progressive disclosure
- YAML frontmatter
- 500-line rule

---

## System Overview

### Two-Hook Architecture

**1. UserPromptSubmit Hook** (Proactive Suggestions)
- **File**: `.claude/hooks/skill-activation-prompt.ts`
- **Trigger**: BEFORE Claude sees user's prompt
- **Purpose**: Suggest relevant skills based on keywords + intent patterns
- **Method**: Injects formatted reminder as context (stdout → Claude's input)
- **Use Cases**: Topic-based skills, implicit work detection

**2. PostToolUse Hook** (Context Tracking)
- **File**: `.claude/hooks/post-tool-use-tracker.ts`
- **Trigger**: AFTER tool use completes
- **Purpose**: Track file changes, build commands, context management
- **Method**: Analyzes edited files, generates relevant commands
- **Use Cases**: Build tracking, repository detection, TSC validation

### Configuration File

**Location**: `.claude/skills/skill-rules.json`

Defines:
- All skills and their trigger conditions
- Enforcement levels (suggest, warn)
- File path patterns (glob)
- Content detection patterns (regex)
- Priority levels (critical, high, medium, low)

---

## Skill Types

### 1. Domain Skills

**Purpose:** Provide comprehensive guidance for specific areas

**Characteristics:**
- Type: `"domain"`
- Enforcement: `"suggest"`
- Priority: `"high"` or `"medium"`
- Advisory, not mandatory
- Topic or domain-specific
- Comprehensive documentation

**Examples:**
- `backend-dev-guidelines` - Node.js/Express/TypeScript patterns
- `frontend-dev-guidelines` - React/TypeScript best practices
- `error-tracking` - Sentry integration guidance

**When to Use:**
- Complex systems requiring deep knowledge
- Best practices documentation
- Architectural patterns
- How-to guides

---

## Quick Start: Creating a New Skill

### Step 1: Create Skill File

**Location:** `.claude/skills/{skill-name}/SKILL.md`

**Template:**
```markdown
---
name: my-new-skill
description: Brief description including keywords that trigger this skill. Mention topics, file types, and use cases. Be explicit about trigger terms.
---

# My New Skill

## Purpose
What this skill helps with

## When to Use
Specific scenarios and conditions

## Key Information
The actual guidance, documentation, patterns, examples
```

**Best Practices:**
- ✅ **Name**: Lowercase, hyphens, descriptive
- ✅ **Description**: Include ALL trigger keywords/phrases (max 1024 chars)
- ✅ **Content**: Under 500 lines - use reference files for details
- ✅ **Examples**: Real code examples
- ✅ **Structure**: Clear headings, lists, code blocks

### Step 2: Add to skill-rules.json

**Basic Template:**
```json
{
  "my-new-skill": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "medium",
    "promptTriggers": {
      "keywords": ["keyword1", "keyword2"],
      "intentPatterns": ["(create|add).*?something"]
    },
    "pathPatterns": ["**/src/**/*.ts", "**/components/**"]
  }
}
```

### Step 3: Test Triggers

**Test UserPromptSubmit:**
```bash
cd .claude/hooks
echo '{"session_id":"test","prompt":"your test prompt"}' | npx tsx skill-activation-prompt.ts
```

Expected output: Skill suggestions if keywords match

### Step 4: Refine Patterns

Based on testing:
- Add missing keywords
- Refine intent patterns to reduce false positives
- Adjust file path patterns
- Test with multiple scenarios

### Step 5: Follow Anthropic Best Practices

✅ Keep SKILL.md under 500 lines
✅ Use progressive disclosure with reference files
✅ Add table of contents to reference files > 100 lines
✅ Write detailed description with trigger keywords
✅ Test with 3+ real scenarios before documenting
✅ Iterate based on actual usage

---

## Enforcement Levels

### SUGGEST (Recommended)

- Reminder injected before Claude sees prompt
- Claude is aware of relevant skills
- Not enforced, just advisory
- **Use For**: Domain guidance, best practices, how-to guides

**Example:** Frontend development guidelines

### WARN (Optional)

- Low priority suggestions
- Advisory only, minimal enforcement
- **Use For**: Nice-to-have suggestions, informational reminders

Rarely used—most skills use SUGGEST enforcement.

---

## Trigger Types

### 1. Keywords

Direct topic mentions in user prompts.

```json
{
  "keywords": [
    "backend",
    "api",
    "controller",
    "service",
    "repository"
  ]
}
```

**Best Practices:**
- Use lowercase
- Include synonyms
- Be specific but comprehensive
- Think about how users naturally describe tasks

### 2. Intent Patterns

Implicit action detection using regex.

```json
{
  "intentPatterns": [
    "(create|add|build|implement).*?(route|endpoint|api)",
    "(fix|debug).*?(error|bug)"
  ]
}
```

**Best Practices:**
- Use case-insensitive matching
- Capture common action verbs
- Match natural language patterns
- Test against real user prompts

### 3. Path Patterns

Location-based activation using glob patterns.

```json
{
  "pathPatterns": [
    "**/backend/**",
    "**/api/**",
    "**/src/controllers/**",
    "**/server/**"
  ]
}
```

**Best Practices:**
- Use `**` for recursive matching
- Be flexible with directory names
- Account for different project structures
- Test against actual file paths

---

## skill-rules.json Schema

```json
{
  "skill-name": {
    "type": "domain",
    "enforcement": "suggest" | "warn",
    "priority": "critical" | "high" | "medium" | "low",
    "promptTriggers": {
      "keywords": ["string", "..."],
      "intentPatterns": ["regex", "..."]
    },
    "pathPatterns": ["glob", "..."]
  }
}
```

**Field Descriptions:**

- **type**: Skill category ("domain" for most skills)
- **enforcement**: How strongly to suggest ("suggest" for advisory)
- **priority**: Display order in suggestions
- **promptTriggers**: Keyword and pattern matching
- **pathPatterns**: File location triggers (optional)

---

## Testing Checklist

When creating a new skill, verify:

- [ ] Skill file created in `.claude/skills/{name}/SKILL.md`
- [ ] Proper frontmatter with name and description
- [ ] Entry added to `skill-rules.json`
- [ ] Keywords tested with real prompts
- [ ] Intent patterns tested with variations
- [ ] File path patterns tested with actual files (if applicable)
- [ ] Priority level matches importance
- [ ] No false positives in testing
- [ ] No false negatives in testing
- [ ] JSON syntax validated: `node -e "require('./.claude/skills/skill-rules.json')"`
- [ ] **SKILL.md under 500 lines** ⭐
- [ ] Reference files created if needed
- [ ] Table of contents added to files > 100 lines

---

## Hook System Details

### UserPromptSubmit Hook Flow

1. User enters prompt in Claude Code
2. Hook receives JSON: `{"session_id": "...", "prompt": "...", "working_dir": "..."}`
3. Script loads `skill-rules.json`
4. Matches prompt against keywords and intent patterns
5. Outputs formatted suggestions to stdout
6. Claude receives suggestions as context

### PostToolUse Hook Flow

1. Edit/Write/MultiEdit tool completes
2. Hook receives JSON with tool name and file path
3. Script tracks edited files in cache
4. Detects affected repositories
5. Generates build/TSC commands
6. Updates cache for future reference

---

## Quick Reference Summary

### Create New Skill (5 Steps)

1. Create `.claude/skills/{name}/SKILL.md` with frontmatter
2. Add entry to `.claude/skills/skill-rules.json`
3. Test with `npx tsx` commands
4. Refine patterns based on testing
5. Keep SKILL.md under 500 lines

### Trigger Types

- **Keywords**: Explicit topic mentions
- **Intent**: Implicit action detection
- **File Paths**: Location-based activation

### Enforcement

- **SUGGEST**: Inject context, most common
- **WARN**: Advisory, rarely used

### Anthropic Best Practices

✅ **500-line rule**: Keep SKILL.md under 500 lines
✅ **Progressive disclosure**: Use reference files for details
✅ **Table of contents**: Add to reference files > 100 lines
✅ **Rich descriptions**: Include all trigger keywords (max 1024 chars)
✅ **Test first**: Build 3+ evaluations before extensive documentation

---

## Debugging Skills

### Test Skill Activation

```bash
# Test with specific prompt
cd .claude/hooks
echo '{"session_id":"test","prompt":"create a new api endpoint"}' | npx tsx skill-activation-prompt.ts

# Expected output: Should list matching skills
```

### Common Issues

**Skill Not Activating:**
- Check keywords include terms from user prompt
- Verify intent patterns match action + target
- Ensure JSON syntax is valid
- Check hook is registered in settings.json

**Too Many False Positives:**
- Make keywords more specific
- Refine intent patterns to be more precise
- Consider adding path patterns to narrow scope

**JSON Validation Error:**
- Run: `node -e "require('./.claude/skills/skill-rules.json')"`
- Check for trailing commas, missing quotes, invalid syntax

---

## Examples

### Backend Domain Skill

```json
{
  "backend-dev-guidelines": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "high",
    "promptTriggers": {
      "keywords": [
        "backend",
        "api",
        "endpoint",
        "controller",
        "service",
        "repository",
        "prisma",
        "database"
      ],
      "intentPatterns": [
        "(create|add|build|implement).*?(route|endpoint|api|controller|service)",
        "(setup|configure).*?(database|prisma)"
      ]
    },
    "pathPatterns": [
      "**/backend/**",
      "**/api/**",
      "**/server/**",
      "**/src/controllers/**",
      "**/src/services/**",
      "**/src/repositories/**"
    ]
  }
}
```

### Frontend Domain Skill

```json
{
  "frontend-dev-guidelines": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "high",
    "promptTriggers": {
      "keywords": [
        "frontend",
        "react",
        "component",
        "ui",
        "interface",
        "page",
        "suspense",
        "mui"
      ],
      "intentPatterns": [
        "(create|add|build|implement).*?(component|page|ui|interface)",
        "(fix|refactor).*?(component|page)"
      ]
    },
    "pathPatterns": [
      "**/frontend/**",
      "**/client/**",
      "**/src/components/**",
      "**/src/pages/**",
      "**/src/features/**"
    ]
  }
}
```

---

## Best Practices Summary

### ✅ DO

- Keep SKILL.md under 500 lines
- Use descriptive, keyword-rich descriptions
- Include real code examples
- Test with multiple scenarios
- Use progressive disclosure for complex topics
- Make keywords comprehensive but specific
- Use intent patterns for implicit actions
- Add path patterns for location-based triggers
- Validate JSON syntax
- Iterate based on actual usage

### ❌ DON'T

- Exceed 500 lines in SKILL.md
- Use vague or generic keywords
- Skip testing before deployment
- Use overly broad intent patterns
- Forget to validate JSON
- Mix concerns in single skill
- Use blocking enforcement unnecessarily
- Skip documentation
- Ignore false positives/negatives

---

## File Locations

**Skills:**
- `.claude/skills/{skill-name}/SKILL.md` - Main skill file
- `.claude/skills/skill-rules.json` - Activation configuration

**Hooks:**
- `.claude/hooks/skill-activation-prompt.sh` - Bash wrapper
- `.claude/hooks/skill-activation-prompt.ts` - TypeScript implementation
- `.claude/hooks/post-tool-use-tracker.sh` - Bash wrapper
- `.claude/hooks/post-tool-use-tracker.ts` - TypeScript implementation
- `.claude/hooks/types/hook-types.ts` - Shared types

**Configuration:**
- `.claude/settings.json` - Hook registration

**Cache:**
- `.claude/hooks/state/` - Session tracking
- `.claude/tsc-cache/` - Build tracking

---

**Skill Status**: COMPLETE ✅
**Version**: 1.0.0
**Last Updated**: 2025-11-01
