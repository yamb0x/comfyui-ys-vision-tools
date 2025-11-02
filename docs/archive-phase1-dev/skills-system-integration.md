# Skills System Integration Documentation

**Last Updated**: 2025-11-01
**Source**: [claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase)

---

## Overview

This project integrates a comprehensive **skills system** that automatically suggests relevant development guidelines based on your prompts and file context. The system uses hooks to monitor your activity and proactively load domain-specific best practices.

### What Are Skills?

Skills are **markdown-based documentation files** that contain:
- Best practices and patterns
- Code examples and templates
- Anti-patterns to avoid
- Quick reference guides
- Checklists for common tasks

Skills are **automatically activated** when you:
- Use specific keywords in prompts
- Edit files in certain directories
- Work on particular types of tasks

---

## Integrated Skills

### 1. **backend-dev-guidelines**

**Purpose**: Node.js/Express/TypeScript backend development patterns

**Activates When**:
- Keywords: backend, api, endpoint, controller, service, repository, prisma, database
- File paths: `**/backend/**`, `**/api/**`, `**/server/**`, `**/src/controllers/**`
- Actions: Creating routes, implementing services, database operations

**Key Principles**:
- Routes only route (delegate to controllers)
- All controllers extend BaseController
- All errors captured to Sentry
- Use unifiedConfig (never process.env)
- Validate all input with Zod
- Repository pattern for data access
- Comprehensive testing required

**File**: `.claude/skills/backend-dev-guidelines/SKILL.md`

---

### 2. **frontend-dev-guidelines**

**Purpose**: React/TypeScript frontend development patterns

**Activates When**:
- Keywords: frontend, react, component, ui, page, suspense, mui, hook
- File paths: `**/frontend/**`, `**/client/**`, `**/src/components/**`, `**/src/pages/**`
- Actions: Creating components, implementing UI, data fetching

**Key Principles**:
- Lazy load heavy components (DataGrid, charts, editors)
- Use Suspense for loading states (no early returns)
- useSuspenseQuery for data fetching
- Feature-based organization
- Style organization (inline <100 lines, separate file >100 lines)
- Import aliases for clean imports
- Performance optimization (useMemo, useCallback, React.memo)

**File**: `.claude/skills/frontend-dev-guidelines/SKILL.md`

---

### 3. **skill-developer**

**Purpose**: Meta-skill for creating and managing skills

**Activates When**:
- Keywords: skill, skill-rules, activation, trigger, hook
- Actions: Creating skills, modifying triggers, debugging activation

**Key Principles**:
- Keep SKILL.md under 500 lines
- Use progressive disclosure with reference files
- Include all trigger keywords in description
- Test with 3+ real scenarios
- Use descriptive naming (lowercase, hyphens)

**File**: `.claude/skills/skill-developer/SKILL.md`

---

### 4. **error-tracking**

**Purpose**: Sentry v8 error tracking and monitoring

**Activates When**:
- Keywords: error, sentry, tracking, monitoring, exception, logging
- File paths: `**/middleware/**`, `**/controllers/**`, `**/services/**`
- Actions: Implementing error handling, setting up Sentry

**Core Principle**: **ALL ERRORS MUST BE CAPTURED TO SENTRY - No exceptions**

**Key Patterns**:
- Controllers extend BaseController for automatic error capture
- Capture with context (tags, user, extra data)
- Use appropriate severity levels
- Add breadcrumbs for error trail
- Filter sensitive data before sending

**File**: `.claude/skills/error-tracking/SKILL.md`

---

## How Auto-Activation Works

### Two-Hook System

#### 1. UserPromptSubmit Hook

**Trigger**: Runs BEFORE Claude sees your prompt

**Purpose**: Suggests relevant skills based on:
- Keywords in your prompt
- Intent patterns (e.g., "create a controller")

**Implementation**: `.claude/hooks/skill-activation-prompt.ts`

**Example**:
```
User: "Help me create a new API endpoint"
Hook: Detects keywords "api" and "endpoint"
Output: Suggests backend-dev-guidelines skill
Claude: Receives suggestion and loads skill if relevant
```

#### 2. PostToolUse Hook

**Trigger**: Runs AFTER Edit/Write/MultiEdit operations

**Purpose**: Tracks context for better assistance:
- Logs edited files with timestamps
- Detects affected repositories
- Generates build/TSC commands
- Updates cache for session continuity

**Implementation**: `.claude/hooks/post-tool-use-tracker.ts`

**Cache Location**: `.claude/tsc-cache/`

---

## Configuration

### skill-rules.json

Central configuration defining all skills and their triggers.

**Location**: `.claude/skills/skill-rules.json`

**Structure**:
```json
{
  "skill-name": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "high",
    "promptTriggers": {
      "keywords": ["keyword1", "keyword2"],
      "intentPatterns": ["(create|add).*?(thing)"]
    },
    "pathPatterns": ["**/directory/**"]
  }
}
```

**Fields**:
- **type**: Skill category ("domain" for most)
- **enforcement**: "suggest" (advisory) or "warn" (low priority)
- **priority**: Display order (critical, high, medium, low)
- **promptTriggers**: Keyword and pattern matching
- **pathPatterns**: File location triggers (optional)

---

## Adding New Skills

### Step 1: Create Skill File

**Location**: `.claude/skills/{skill-name}/SKILL.md`

**Template**:
```markdown
---
name: my-skill
description: Detailed description with ALL trigger keywords
---

# My Skill Title

## Purpose
What this skill helps with

## When to Use
Specific scenarios

## Content
Guidelines, examples, best practices
```

### Step 2: Update skill-rules.json

Add entry to `.claude/skills/skill-rules.json`:

```json
{
  "my-skill": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "medium",
    "promptTriggers": {
      "keywords": ["keyword1", "keyword2"],
      "intentPatterns": ["pattern.*?here"]
    },
    "pathPatterns": ["**/path/**"]
  }
}
```

### Step 3: Test Activation

```bash
cd .claude/hooks
echo '{"session_id":"test","prompt":"your test prompt"}' | npx tsx skill-activation-prompt.ts
```

### Step 4: Refine

Based on testing:
- Add missing keywords
- Adjust intent patterns
- Update path patterns
- Test with variations

---

## Testing Skills

### Manual Hook Test

```bash
# Test skill activation
cd .claude/hooks
echo '{"session_id":"test","prompt":"create a new api endpoint"}' | npx tsx skill-activation-prompt.ts

# Expected output: Suggests backend-dev-guidelines
```

### Validate JSON

```bash
# Check skill-rules.json syntax
node -e "require('./.claude/skills/skill-rules.json')"
```

### Real-World Test

1. Edit a file matching pathPatterns
2. Use prompt with trigger keywords
3. Verify skill suggestions appear

---

## Troubleshooting

### Skill Not Activating

**Check**:
- ✅ Keywords include terms from your prompt
- ✅ Intent patterns match your action + target
- ✅ JSON syntax is valid
- ✅ Hooks are registered in settings.json
- ✅ Hook files are executable (chmod +x)

**Debug**:
```bash
cd .claude/hooks
echo '{"prompt":"your prompt"}' | npx tsx skill-activation-prompt.ts
```

### Too Many False Positives

**Solutions**:
- Make keywords more specific
- Refine intent patterns to be more precise
- Add path patterns to narrow scope

### JSON Syntax Error

**Validate**:
```bash
node -e "require('./.claude/skills/skill-rules.json')"
```

Check for:
- Trailing commas
- Missing quotes
- Unmatched brackets

---

## File Structure

```
.claude/
├── skills/                          # Skills directory
│   ├── backend-dev-guidelines/
│   │   └── SKILL.md
│   ├── frontend-dev-guidelines/
│   │   └── SKILL.md
│   ├── skill-developer/
│   │   └── SKILL.md
│   ├── error-tracking/
│   │   └── SKILL.md
│   └── skill-rules.json             # Activation config
├── hooks/                           # Hooks directory
│   ├── skill-activation-prompt.sh   # Bash wrapper
│   ├── skill-activation-prompt.ts   # TypeScript impl
│   ├── post-tool-use-tracker.sh     # Bash wrapper
│   ├── post-tool-use-tracker.ts     # TypeScript impl
│   ├── package.json                 # Hook dependencies
│   ├── types/
│   │   └── hook-types.ts            # Shared types
│   └── state/                       # Session tracking
├── tsc-cache/                       # Build tracking cache
│   ├── edited-files.log
│   ├── affected-repos.txt
│   └── commands.txt
└── settings.json                    # Hook registration
```

---

## Customization

### Adjust Path Patterns

Update `skill-rules.json` to match your project structure:

```json
{
  "backend-dev-guidelines": {
    "pathPatterns": [
      "**/your-backend-dir/**",
      "**/custom-api-dir/**"
    ]
  }
}
```

### Change Priority

Modify skill priority to control display order:

```json
{
  "my-skill": {
    "priority": "critical"  // critical > high > medium > low
  }
}
```

### Disable Skill

Remove from `skill-rules.json` or set enforcement to "warn":

```json
{
  "my-skill": {
    "enforcement": "warn"  // Lower priority
  }
}
```

---

## Best Practices

### ✅ DO

- Keep SKILL.md files under 500 lines
- Include all relevant keywords in descriptions
- Test skills with multiple scenarios before deploying
- Use specific keywords to reduce false positives
- Update path patterns to match project structure
- Validate JSON syntax after changes

### ❌ DON'T

- Use overly generic keywords
- Skip testing before deployment
- Exceed 500 lines in SKILL.md
- Use blocking enforcement (not implemented in this integration)
- Ignore false positives/negatives
- Forget to make hooks executable

---

## Dependencies

**Required**:
- Node.js (for tsx execution)
- npm/npx (for package management)

**Installed**:
- tsx@^4.19.0 (TypeScript execution)
- @types/node@^22.0.0 (Node.js types)

**Location**: `.claude/hooks/package.json`

---

## Performance

**Hook Execution Time**:
- UserPromptSubmit: <100ms
- PostToolUse: <50ms

**Impact**: Minimal - hooks run asynchronously and don't block Claude's response

**Cache**: PostToolUse hook maintains cache in `.claude/tsc-cache/` for session continuity

---

## References

- **Source Repository**: https://github.com/diet103/claude-code-infrastructure-showcase
- **Integration Guide**: https://github.com/diet103/claude-code-infrastructure-showcase/blob/main/CLAUDE_INTEGRATION_GUIDE.md
- **Anthropic Best Practices**: 500-line rule, progressive disclosure, YAML frontmatter

---

## Support

For issues or questions:

1. Check troubleshooting section above
2. Validate JSON syntax
3. Test hooks manually
4. Review skill-rules.json configuration
5. Consult source repository documentation

---

**Integration Status**: ✅ COMPLETE
**Skills**: 4 installed and configured
**Hooks**: 2 active (UserPromptSubmit, PostToolUse)
**Auto-Activation**: Enabled
