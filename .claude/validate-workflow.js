#!/usr/bin/env node

/**
 * Workflow Validation Script
 * Checks if Claude is following project-specific CLAUDE.md rules
 */

const fs = require('fs');
const path = require('path');

function validateWorkflow() {
  console.log('🔍 Validating Claude Code workflow compliance...\n');
  
  // Check if CLAUDE.md exists
  const claudeMdPath = path.join(process.cwd(), 'CLAUDE.md');
  if (!fs.existsSync(claudeMdPath)) {
    console.log('✅ No CLAUDE.md found - using default workflow');
    return true;
  }
  
  console.log('📋 CLAUDE.md detected - project-specific rules apply!');
  console.log('');
  
  // Check for tasks directory
  const tasksDir = path.join(process.cwd(), '.claude', 'tasks');
  const taskFiles = fs.existsSync(tasksDir) ? fs.readdirSync(tasksDir).filter(f => f.endsWith('.md')) : [];
  
  console.log('📂 Task Management Status:');
  console.log(`   Tasks directory: ${fs.existsSync(tasksDir) ? '✅ Found' : '❌ Missing'}`);
  console.log(`   Task files: ${taskFiles.length} found`);
  
  if (taskFiles.length > 0) {
    console.log('   Recent tasks:');
    taskFiles.slice(-3).forEach(file => {
      const stats = fs.statSync(path.join(tasksDir, file));
      console.log(`   - ${file} (${stats.mtime.toLocaleDateString()})`);
    });
  }
  
  console.log('');
  console.log('⚠️  MANDATORY WORKFLOW REMINDERS:');
  console.log('');
  console.log('   1. 📝 Create detailed plan in .claude/tasks/TASK_NAME.md FIRST');
  console.log('   2. 🤝 Get approval before proceeding with implementation');
  console.log('   3. 📊 Use TodoWrite tool for progress tracking');
  console.log('   4. 🔄 Update task status: pending → in_progress → completed');
  console.log('   5. 📄 Update context files after completion');
  console.log('');
  console.log('🚨 VIOLATION ALERT: Starting work without following these steps');
  console.log('    violates the established project workflow protocol!');
  console.log('');
  
  return true;
}

if (require.main === module) {
  validateWorkflow();
}

module.exports = validateWorkflow;