# Task Plan: Integrate Claude Code Skills System

**Date**: 2025-11-01
**Status**: Planning Phase
**Source**: https://github.com/diet103/claude-code-infrastructure-showcase

---

## Objective

Integrate the skills system from claude-code-infrastructure-showcase into this template project, including:
- 4 specific skills (backend-dev-guidelines, frontend-dev-guidelines, skill-developer, error-tracking)
- Auto-activation system with skill-rules.json
- Essential hooks for skill activation
- Documentation for the integration

---

## Implementation Plan

### Phase 1: Directory Structure Setup

**Create necessary directories:**
```
.claude/
├── skills/                    # NEW - Skills directory
│   ├── backend-dev-guidelines/
│   ├── frontend-dev-guidelines/
│   ├── skill-developer/
│   └── error-tracking/
└── hooks/                     # NEW - Hooks directory
    ├── state/                 # NEW - Session tracking
    └── types/                 # NEW - TypeScript types
```

**Reasoning**: Following the showcase repository structure for compatibility

---

### Phase 2: Install Required Dependencies

**Add TypeScript execution tools:**
- tsx (for executing TypeScript hooks)
- @types/node (for Node.js type definitions)

**Installation Method**: Check if package.json exists in project root
- If yes: Add to devDependencies
- If no: Create minimal package.json for hook execution

**Reasoning**: Hooks use TypeScript (.ts files) and need tsx to execute

---

### Phase 3: Create Skill Files

**3.1 Backend Dev Guidelines Skill**
- Create `.claude/skills/backend-dev-guidelines/SKILL.md`
- Content: Node.js/Express/TypeScript patterns
- Focus: Layered architecture, BaseController, Prisma, Sentry, Zod validation

**3.2 Frontend Dev Guidelines Skill**
- Create `.claude/skills/frontend-dev-guidelines/SKILL.md`
- Content: React/TypeScript/MUI v7 patterns
- Focus: Suspense, lazy loading, useSuspenseQuery, file organization

**3.3 Skill Developer Skill**
- Create `.claude/skills/skill-developer/SKILL.md`
- Content: Creating and managing skills, skill-rules.json, hooks
- Focus: Skill structure, YAML frontmatter, trigger patterns

**3.4 Error Tracking Skill**
- Create `.claude/skills/error-tracking/SKILL.md`
- Content: Sentry v8 integration patterns
- Focus: Error capture, breadcrumbs, user context, performance monitoring

**Customization**: Adapt tech stack references to match this template project
- Check if project uses Express/Prisma (backend) or React/MUI (frontend)
- Adjust examples if different framework is used
- Keep architecture patterns even if tech differs

---

### Phase 4: Create skill-rules.json Configuration

**Create**: `.claude/skills/skill-rules.json`

**Configuration Structure**:
```json
{
  "backend-dev-guidelines": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "high",
    "promptTriggers": {
      "keywords": ["backend", "api", "express", "controller", "service", "repository", "prisma"],
      "intentPatterns": ["(create|add|build|implement).*?(route|endpoint|api|controller|service)"]
    },
    "pathPatterns": ["**/backend/**", "**/api/**", "**/server/**", "**/src/controllers/**"]
  },
  "frontend-dev-guidelines": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "high",
    "promptTriggers": {
      "keywords": ["frontend", "react", "component", "ui", "interface", "suspense", "mui"],
      "intentPatterns": ["(create|add|build|implement).*?(component|page|ui|interface)"]
    },
    "pathPatterns": ["**/frontend/**", "**/client/**", "**/src/components/**", "**/src/pages/**"]
  },
  "skill-developer": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "medium",
    "promptTriggers": {
      "keywords": ["skill", "skill-rules", "activation", "trigger", "hook"],
      "intentPatterns": ["(create|add|modify|debug).*?skill"]
    }
  },
  "error-tracking": {
    "type": "domain",
    "enforcement": "suggest",
    "priority": "medium",
    "promptTriggers": {
      "keywords": ["error", "sentry", "tracking", "monitoring", "exception", "logging"],
      "intentPatterns": ["(add|implement|setup).*?(error|sentry|tracking|monitoring)"]
    },
    "pathPatterns": ["**/middleware/**", "**/controllers/**", "**/services/**"]
  }
}
```

**Customization Required**:
- Update `pathPatterns` to match actual project directory structure
- Will need to explore project to determine correct paths

---

### Phase 5: Create Hook Files

**5.1 Skill Activation Hook (UserPromptSubmit)**

**Files to create**:
- `.claude/hooks/skill-activation-prompt.sh` (bash wrapper)
- `.claude/hooks/skill-activation-prompt.ts` (TypeScript implementation)

**Functionality**:
- Reads stdin JSON with user prompt
- Loads skill-rules.json
- Matches keywords and intent patterns
- Outputs formatted skill suggestions to stdout

**5.2 Post-Tool-Use Tracker Hook**

**Files to create**:
- `.claude/hooks/post-tool-use-tracker.sh` (bash wrapper)
- `.claude/hooks/post-tool-use-tracker.ts` (TypeScript implementation)

**Functionality**:
- Tracks edited files after Edit/Write/MultiEdit operations
- Detects affected repositories
- Generates build/TSC commands
- Caches in `.claude/tsc-cache/`

**5.3 Hook Types and Interfaces**

**File to create**: `.claude/hooks/types/hook-types.ts`

**Contents**:
- HookInput interface
- SkillRule interface
- PromptTriggers interface
- Shared TypeScript types for hooks

---

### Phase 6: Update settings.json

**Modify**: `.claude/settings.json`

**Add hook registrations**:
```json
{
  "hooks": {
    "UserPromptSubmit": [
      "./.claude/hooks/skill-activation-prompt.sh"
    ],
    "PostToolUse": [
      "./.claude/hooks/post-tool-use-tracker.sh"
    ]
  }
}
```

**Note**: Preserve any existing hooks configuration

---

### Phase 7: Make Hooks Executable

**Commands**:
```bash
chmod +x .claude/hooks/skill-activation-prompt.sh
chmod +x .claude/hooks/post-tool-use-tracker.sh
```

**Critical**: Windows may not require this, but Linux/Mac do
- Check platform and adjust accordingly

---

### Phase 8: Create Documentation

**8.1 Integration Documentation**

**Create**: `docs/skills-system-integration.md`

**Contents**:
- Overview of skills system
- What was integrated and why
- How auto-activation works
- How to add new skills
- How to test skill activation
- Troubleshooting guide

**8.2 Update CLAUDE.md**

**Add section**: "Skills System"

**Contents**:
- Brief overview of available skills
- Reference to detailed documentation
- Quick start guide

---

### Phase 9: Testing & Validation

**9.1 Test Hook Execution**

```bash
# Test skill activation hook
echo '{"session_id":"test","prompt":"create a new api endpoint"}' | npx tsx .claude/hooks/skill-activation-prompt.ts

# Expected: Should suggest backend-dev-guidelines skill
```

**9.2 Test Skill Activation**

- Edit a file matching pathPatterns (e.g., controllers/test.ts)
- Use prompt with trigger keywords (e.g., "create a controller")
- Verify skill suggestions appear

**9.3 Verify File Structure**

- All skill files exist in correct locations
- skill-rules.json is valid JSON
- Hooks are executable
- Dependencies installed

---

## Success Criteria

✅ 4 skills installed in `.claude/skills/`
✅ `skill-rules.json` created with activation patterns
✅ 2 essential hooks installed and executable
✅ Hooks registered in `settings.json`
✅ Dependencies (tsx, @types/node) installed
✅ Documentation created in `docs/`
✅ CLAUDE.md updated with skills reference
✅ Skill activation tested and working
✅ Path patterns customized for this project

---

## Risk Assessment

**Low Risk**:
- Skills are documentation-only (no code execution)
- Hooks run in isolated process
- Can be disabled via env vars

**Medium Risk**:
- Path patterns may need adjustment for project structure
- Tech stack differences may require skill customization

**Mitigation**:
- Test hooks before registering in settings.json
- Start with one skill, verify, then add others
- Keep original showcase repository link for reference

---

## Timeline Estimate

- Phase 1-2: 5 minutes (directory structure, dependencies)
- Phase 3-4: 15 minutes (skill files, skill-rules.json)
- Phase 5-6: 10 minutes (hooks, settings update)
- Phase 7-9: 10 minutes (permissions, docs, testing)

**Total**: ~40 minutes

---

## Notes

- This is an MVP integration focusing on core functionality
- Additional resource files from showcase can be added later if needed
- Skills can be customized further based on actual project tech stack
- Hook functionality can be extended with additional features

---

## Next Steps After Approval

1. Create directory structure
2. Install dependencies
3. Create skill files (4 files)
4. Create skill-rules.json
5. Create hook files (6 files total)
6. Update settings.json
7. Make hooks executable
8. Create documentation (2 files)
9. Test integration
10. Mark all tasks complete

**Ready for approval to proceed with implementation.**
