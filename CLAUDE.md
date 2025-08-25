# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

This is a Hebrew audio transcription service that uses OpenAI's Whisper API. The project is being enhanced with WhisperX for local processing and speaker diarization capabilities.

## Development Commands

### Quick Reference
- `python transcribe.py <audio_file>` - Transcribe an audio file
- `pip install -r requirements.txt` - Install dependencies

## Feature Development Process

### New Feature Specification Process

When planning new features, follow this structured approach using the `docs/specs/` directory.

**Note:** For detailed specification writing instructions, refer to `.claude/specs-instruction.md` which contains the complete templates and guidelines from the agente-ai project.

#### 1. Feature Planning Conversation Flow

**Initial Discussion**:
- Start with: "I want to work on a new feature: [feature name]"
- Discuss high-level goals and user needs
- Identify which users will benefit
- Determine scope and constraints

**Requirements Gathering**:
- Create `requirements.md` with user stories and acceptance criteria
- Use format: "As a [user type], I want [functionality], so that [benefit]"
- Include detailed acceptance criteria using EARS notation
- Consider edge cases and integration points

**Design Planning**:
- Create `design.md` with technical architecture
- Define component hierarchy and data flow
- Specify interfaces and data models
- Include security and performance considerations

**Task Breakdown**:
- Create `tasks.md` with implementation plan
- Break down into concrete, testable tasks
- Use status icons: ✅ (completed), ⏳ (in progress), [ ] (pending)
- Include dependencies and priorities

#### 2. Feature Specification Directory Structure

```
docs/specs/
├── [feature-name]/
│   ├── requirements.md     # User stories and acceptance criteria
│   ├── design.md          # Technical architecture and design decisions
│   └── tasks.md           # Implementation breakdown and progress tracking
```

#### 3. Requirements Template

Use this template for `requirements.md`:

```markdown
# Requirements Document

## Introduction
[Brief description of the feature and its purpose]

## Requirements

### Requirement 1: [Feature Name]
**User Story:** As a [user type], I want [functionality], so that [benefit].

#### Acceptance Criteria
Use EARS notation for clear, testable requirements:
1. WHEN [condition/event] THEN the system SHALL [expected behavior]
2. WHEN [condition/event] THEN the system SHALL [expected behavior]
[Continue with numbered criteria...]

### Requirement 2: [Feature Name]
[Continue with additional requirements...]
```

##### EARS Notation Examples:
- **Event-driven**: WHEN a user uploads a file THEN the system SHALL validate the format
- **State-driven**: WHILE processing is active THEN the system SHALL display progress
- **Optional**: WHERE supported THEN the system SHALL use GPU acceleration
- **Complex**: WHEN file size exceeds 25MB AND local processing is enabled THEN the system SHALL use chunked processing

#### 4. Design Template

Use this template for `design.md`:

```markdown
# Design Document

## Overview
[High-level description of the feature architecture]

## Architecture
[Technical architecture diagrams and component relationships]
- Use ASCII diagrams for architecture visualization
- Show data flow and component interactions

## Components and Interfaces
[Define interfaces, data structures, and APIs]
- Use TypeScript/Python type definitions
- Specify input/output contracts
- Document error responses

## Data Models
[Database schema changes and data structures]

## Error Handling
[Error scenarios and handling strategies]
- Define error categories
- Specify recovery mechanisms
- Include fallback strategies

## Testing Strategy
[Testing approach and coverage requirements]
- Unit test requirements
- Integration test scenarios
- Performance benchmarks

## Security Considerations
[Security implications and protection measures]

## Performance Optimization
[Performance considerations and optimization strategies]
```

#### 5. Tasks Template

Use this template for `tasks.md`:

```markdown
# Implementation Plan

## Current Status
**Branch:** [branch-name]
**Progress:** [X/Y tasks completed] ([percentage]%)
**Current Focus:** [Current priority]

### Completed Tasks:
- ✅ [Task description with completion date]

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

#### 6. Status Icons Convention

Use these icons consistently in tasks.md:
- ✅ Completed task
- ⏳ In progress task
- [ ] Pending task
- ❌ Blocked or cancelled task
- ⚠️ Task with issues or warnings

#### 7. Best Practices

**Requirements Phase**:
- Focus on user value and business needs
- Use EARS notation for clear, testable acceptance criteria
- Consider all user types and edge cases
- Reference existing similar features
- Number all requirements for easy reference

**Design Phase**:
- Prioritize component reuse and consistency
- Follow existing architectural patterns
- Consider security and performance implications
- Design for testability and maintainability
- Include clear interface definitions

**Implementation Phase**:
- Update task progress regularly (mark completed immediately)
- Run quality checks frequently
- Document architectural decisions
- Track implementation with specific dates
- Use modular approach - create separate specs for different features

**Organization**:
- Store specs in project repository for version control
- Create multiple focused specs instead of one massive specification
- Keep specs alongside code for maintainability
- Update specs as implementation evolves

#### 8. Example Spec Creation Flow

When asked to create a specification:

1. **Always create three documents**:
   - `requirements.md` - User stories with EARS notation
   - `design.md` - System architecture and technical design
   - `tasks.md` - Implementation plan with trackable tasks

2. **Follow the templates exactly** as shown above

3. **Use proper formatting**:
   - EARS notation for requirements
   - ASCII diagrams for architecture
   - Status icons for task tracking
   - Clear section headers and numbering

4. **Include all sections** even if briefly:
   - Don't skip sections in templates
   - Mark sections as "N/A" if truly not applicable
   - Provide reasoning for any omissions

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints for function signatures
- Keep functions focused and under 50 lines
- Use descriptive variable names
- Add docstrings for all public functions

### Error Handling
- Always handle specific exceptions
- Provide helpful error messages
- Include recovery suggestions
- Log errors appropriately
- Never silently fail

### Testing
- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Test edge cases and error conditions
- Mock external dependencies

## Project-Specific Patterns

### Audio Processing
- Always validate file format before processing
- Check file size limits
- Handle audio loading errors gracefully
- Clean up temporary files

### Transcription Service
- Maintain backward compatibility with existing CLI
- Support both local and API-based processing
- Provide clear progress indicators
- Handle long-running operations properly

### Hebrew Language Support
- Ensure proper RTL text handling
- Test with Hebrew audio samples
- Validate character encoding
- Support mixed language content

## Important Notes

- **Specification Templates**: See `.claude/specs-instruction.md` for complete specification templates and guidelines
- **Specification First**: Always create full specifications before implementation
- **Three Documents Rule**: Every feature needs requirements.md, design.md, and tasks.md
- **EARS Notation**: Use WHEN/THEN format for all acceptance criteria (see examples in `.claude/specs-instruction.md`)
- **Progress Tracking**: Update tasks.md immediately when completing tasks
- **Status Icons**: Use ✅, ⏳, [ ] consistently for task status
- **User Stories**: Start every requirement with "As a [user], I want [feature], so that [benefit]"
- **Architecture Diagrams**: Include ASCII diagrams in design.md
- **Modular Specs**: Create separate spec folders for different features