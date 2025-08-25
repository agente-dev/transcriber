# Feature Specification Writing Instructions

## Overview
This document provides comprehensive guidance for creating structured feature specifications. These instructions ensure consistent, thorough planning and documentation across all feature development.

## Feature Specification Process

### 1. Directory Structure
Create specs in `docs/specs/` with this structure:
```
docs/specs/
├── [feature-name]/
│   ├── requirements.md     # User stories and acceptance criteria
│   ├── design.md          # Technical architecture and design decisions
│   └── tasks.md           # Implementation breakdown and progress tracking
```

### 2. Requirements Document Template

Use EARS (Easy Approach to Requirements Syntax) notation for clear, testable requirements:

```markdown
# Requirements Document

## Introduction
[Brief description of the feature and its purpose]

## Requirements

### Requirement 1
**User Story:** As a [user type], I want [functionality], so that [benefit].

#### Acceptance Criteria
Use EARS notation:
1. WHEN [condition/event] THE SYSTEM SHALL [expected behavior]
2. WHEN [condition/event] THE SYSTEM SHALL [expected behavior]

#### Example EARS Format:
- **Event-driven**: WHEN a user uploads a file THEN the system SHALL validate file format
- **State-driven**: WHILE processing is active THEN the system SHALL display progress
- **Optional**: WHERE GPU is available THEN the system SHALL use acceleration
- **Complex**: WHEN file size exceeds 25MB AND local processing is enabled THEN the system SHALL use chunked processing
```

### 3. Design Document Template

```markdown
# Design Document

## Overview
[High-level description of the feature architecture]

## Architecture
[Technical architecture diagrams and component relationships]

## Components and Interfaces
[TypeScript interfaces and component specifications]

## Data Models
[Database schema changes and data structures]

## Error Handling
[Error scenarios and handling strategies]

## Testing Strategy
[Testing approach and coverage requirements]

## Security Considerations
[Security implications and protection measures]

## Performance Optimization
[Performance considerations and optimization strategies]
```

### 4. Tasks Document Template

```markdown
# Implementation Plan

## Current Status
**Branch:** [branch-name]
**Progress:** [X/Y tasks completed] ([percentage]%)
**Current Focus:** [Current priority]

### Completed Tasks:
- ✅ [Task description]

### In Progress:
- ⏳ [Task description]

### Implementation Tasks:

## Phase 1: [Phase Name] [X/Y]

- [ ] 1. **[Task Title]**
  - [ ] [Sub-task 1]
  - [ ] [Sub-task 2]
  - _Requirements: [Reference to requirements]_

- [ ] 2. **[Task Title]**
  [Continue with detailed task breakdown...]

## Dependencies Between Tasks
[Document task dependencies and critical path]

## Risk Mitigation Tasks
[Identify risks and mitigation strategies]

## Success Metrics
[Define completion criteria and KPIs]
```

## Conversation Flow for New Features

### Step 1: Requirements Discussion
- User initiates: "I want to work on a new feature: [feature name]"
- Review existing codebase for similar functionality
- Ask clarifying questions about requirements
- Define user stories and acceptance criteria together
- Create `requirements.md`

### Step 2: Design Planning
- Analyze requirements and propose technical approach
- Identify reusable components and patterns
- Define data models and API contracts
- Create `design.md` with architecture decisions

### Step 3: Task Breakdown
- Break down design into concrete implementation tasks
- Identify dependencies and priorities
- Create `tasks.md` with detailed implementation plan
- Set up development environment and branch

### Step 4: Implementation
- Implement tasks following the plan
- Update progress in `tasks.md`
- Run tests and quality checks
- Document completed work

## Best Practices

### Requirements Phase:
- Focus on user value and business needs
- Use EARS notation for clear, testable acceptance criteria
- Consider all user types and edge cases
- Reference existing similar features
- Import existing requirements flexibly (copy from other tools, documents)

### Design Phase:
- Prioritize component reuse and consistency
- Follow existing architectural patterns
- Consider security and performance implications
- Design for testability and maintainability
- Continuously refine specifications as understanding evolves

### Implementation Phase:
- Update task progress regularly
- Run quality checks frequently
- Document architectural decisions
- Track and update tasks by marking completed work
- Use modular approach - create separate specs for different features

### Organization and Collaboration:
- Store specs in project repository for version control
- Create multiple focused specs instead of one massive specification
- Organize specs by feature domain (e.g., authentication, file-processing)
- Keep specs alongside code for maintainability

## Quality Gates

Before moving between phases:
- **Requirements → Design**: All user stories have clear acceptance criteria
- **Design → Implementation**: All components and APIs are defined
- **Implementation → Completion**: All tests pass and documentation is updated

## Key Principles

1. **Always Search First**: Before creating new components, search for existing functionality
2. **Reuse Over Create**: Leverage existing components, hooks, and patterns
3. **Document Why**: If creating new components, document why existing ones couldn't be reused
4. **Test Coverage**: Maintain minimum 70% branch coverage
5. **Clean Architecture**: Follow single responsibility principle when splitting files
6. **File Size Limits**: Maximum 800 lines for complex modules, 500 lines ideal

## Status Icons Convention

Use these icons consistently in tasks.md:
- ✅ Completed task
- ⏳ In progress task  
- [ ] Pending task
- ❌ Blocked or cancelled task
- ⚠️ Task with issues or warnings

## Important Reminders

1. **Always create three documents**: requirements.md, design.md, and tasks.md
2. **Use EARS notation**: WHEN/THEN format for all acceptance criteria
3. **Include all sections**: Don't skip template sections without justification
4. **Update progress immediately**: Mark tasks completed as soon as done
5. **Be specific**: Include concrete details, not vague descriptions
6. **Reference requirements**: Link tasks back to specific requirements
7. **Consider all users**: Think about different user types and edge cases
8. **Design for reuse**: Check existing components before creating new ones

This structured approach ensures thorough planning, consistent implementation, and maintainable code that integrates well with the existing codebase architecture.