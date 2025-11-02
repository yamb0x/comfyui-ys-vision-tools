#!/usr/bin/env node

/**
 * Post-Tool-Use Tracker Hook
 *
 * This hook runs AFTER Edit/Write/MultiEdit operations and tracks:
 * - Edited files with timestamps
 * - Affected repositories
 * - Build commands to run
 * - TypeScript compilation commands
 *
 * Flow:
 * 1. Read JSON input from stdin (tool name, file path, etc.)
 * 2. Track edited file in cache
 * 3. Detect affected repository
 * 4. Generate build/TSC commands
 * 5. Update cache files
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import type { HookInput } from './types/hook-types.js';

// Get directory paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CLAUDE_DIR = join(__dirname, '..');
const CACHE_DIR = join(CLAUDE_DIR, 'tsc-cache');

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
 * Ensure cache directory exists
 */
function ensureCacheDir() {
  if (!existsSync(CACHE_DIR)) {
    mkdirSync(CACHE_DIR, { recursive: true });
  }
}

/**
 * Get files edited by tool operations
 */
function getEditedFiles(hookInput: HookInput): string[] {
  const files: string[] = [];

  if (hookInput.tool_input?.file_path) {
    files.push(hookInput.tool_input.file_path);
  }

  if (hookInput.tool_input?.paths) {
    files.push(...hookInput.tool_input.paths);
  }

  return files;
}

/**
 * Detect repository from file path
 */
function detectRepository(filePath: string): string | null {
  const lowerPath = filePath.toLowerCase();

  // Skip markdown files
  if (lowerPath.endsWith('.md')) {
    return null;
  }

  // Check for common repository indicators
  if (lowerPath.includes('frontend') || lowerPath.includes('client')) {
    return 'frontend';
  }

  if (lowerPath.includes('backend') || lowerPath.includes('server') || lowerPath.includes('api')) {
    return 'backend';
  }

  if (lowerPath.includes('packages')) {
    // Extract package name from path
    const match = filePath.match(/packages[/\\]([^/\\]+)/);
    return match ? `packages/${match[1]}` : 'packages';
  }

  // Default to root
  return 'root';
}

/**
 * Track edited file in cache
 */
function trackEditedFile(filePath: string, sessionId: string) {
  ensureCacheDir();

  const logFile = join(CACHE_DIR, 'edited-files.log');
  const timestamp = new Date().toISOString();
  const entry = `${timestamp}|${sessionId}|${filePath}\n`;

  appendFileSync(logFile, entry, 'utf-8');
}

/**
 * Update affected repositories cache
 */
function updateAffectedRepos(repos: Set<string>) {
  ensureCacheDir();

  const reposFile = join(CACHE_DIR, 'affected-repos.txt');
  const existing = existsSync(reposFile)
    ? readFileSync(reposFile, 'utf-8').split('\n').filter(Boolean)
    : [];

  const combined = new Set([...existing, ...repos]);
  writeFileSync(reposFile, Array.from(combined).join('\n'), 'utf-8');
}

/**
 * Generate build commands for affected repos
 */
function generateBuildCommands(repos: Set<string>): string[] {
  const commands: string[] = [];

  for (const repo of repos) {
    if (repo === 'frontend') {
      commands.push('cd frontend && npm run build');
      commands.push('cd frontend && npx tsc --noEmit');
    } else if (repo === 'backend') {
      commands.push('cd backend && npm run build');
      commands.push('cd backend && npx tsc --noEmit');
    } else if (repo.startsWith('packages/')) {
      const pkgName = repo.split('/')[1];
      commands.push(`cd packages/${pkgName} && npm run build`);
      commands.push(`cd packages/${pkgName} && npx tsc --noEmit`);
    }
  }

  return commands;
}

/**
 * Update commands cache
 */
function updateCommandsCache(commands: string[]) {
  if (commands.length === 0) return;

  ensureCacheDir();

  const commandsFile = join(CACHE_DIR, 'commands.txt');
  const existing = existsSync(commandsFile)
    ? readFileSync(commandsFile, 'utf-8').split('\n').filter(Boolean)
    : [];

  // Deduplicate commands
  const combined = new Set([...existing, ...commands]);
  writeFileSync(commandsFile, Array.from(combined).join('\n'), 'utf-8');
}

/**
 * Main execution
 */
async function main() {
  try {
    // Read input from stdin
    const input = await readStdin();
    const hookInput: HookInput = JSON.parse(input);

    // Check if this is a tool we track
    const trackedTools = ['Edit', 'Write', 'MultiEdit'];
    if (!hookInput.tool_name || !trackedTools.includes(hookInput.tool_name)) {
      process.exit(0);
    }

    // Get edited files
    const editedFiles = getEditedFiles(hookInput);
    if (editedFiles.length === 0) {
      process.exit(0);
    }

    // Track files and detect repositories
    const affectedRepos = new Set<string>();

    for (const filePath of editedFiles) {
      // Track file
      trackEditedFile(filePath, hookInput.session_id);

      // Detect repository
      const repo = detectRepository(filePath);
      if (repo) {
        affectedRepos.add(repo);
      }
    }

    // Update affected repositories
    if (affectedRepos.size > 0) {
      updateAffectedRepos(affectedRepos);

      // Generate and cache build commands
      const commands = generateBuildCommands(affectedRepos);
      updateCommandsCache(commands);
    }

    process.exit(0);
  } catch (error) {
    console.error('Error in post-tool-use-tracker hook:', error);
    process.exit(1);
  }
}

// Run main
main();
