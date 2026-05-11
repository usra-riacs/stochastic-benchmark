# Naming Convention Refactor for v2.0

## Overview

The current codebase uses non-standard naming conventions that violate PEP 8 guidelines. This document outlines a plan to refactor the naming conventions in a future major version (v2.0) to align with Python best practices.

## Current State

### PEP 8 Violations

The codebase currently uses **PascalCase for method names** instead of the recommended **snake_case**:

**Examples:**
- `initAll()` → should be `init_all()`
- `run_Bootstrap()` → should be `run_bootstrap()`
- `run_Interpolate()` → should be `run_interpolate()`
- `run_Stats()` → should be `run_stats()`
- `StatsSingle()` → should be `stats_single()` (function, not class)
- `get_TrainingResults_recipe()` → should be `get_training_results_recipe()`
- `get_TrainingStats_recipe()` → should be `get_training_stats_recipe()`

### Why This Exists

This appears to be a deliberate design choice made early in the project. The naming convention is:
- Consistent throughout the codebase
- Used in all examples and documentation
- Part of the public API
- Not a bug, just a stylistic deviation from PEP 8

## Why Change It?

### Benefits

1. **PEP 8 Compliance**: Aligns with Python community standards
2. **Better Tooling**: Many linters/formatters expect snake_case for methods
3. **Consistency**: Matches Python standard library conventions
4. **Readability**: snake_case is more readable for longer method names
5. **Professional**: Shows adherence to Python best practices

### Costs

1. **Breaking Change**: Existing code using the library will break
2. **Documentation Update**: All examples, tutorials, and docs need updating
3. **Migration Effort**: Users need to update their code
4. **Testing**: Extensive testing needed to ensure nothing breaks

## Refactor Plan

### Phase 1: Inventory (v1.x)

- [ ] Create comprehensive list of all public API methods with non-standard naming
- [ ] Categorize by module (stochastic_benchmark, bootstrap, stats, etc.)
- [ ] Identify which methods are most commonly used (check examples)
- [ ] Document current usage patterns

### Phase 2: Deprecation Period (v1.x → v2.0)

**Option A: Dual API (Recommended)**
```python
# Add new snake_case methods alongside old PascalCase ones
def init_all(self, ...):
    """New naming convention."""
    # implementation
    
def initAll(self, ...):
    """Deprecated: Use init_all() instead."""
    warnings.warn(
        "initAll() is deprecated and will be removed in v2.0. "
        "Use init_all() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.init_all(...)
```

**Timeline:**
- v1.x: Add new methods, deprecate old ones with warnings
- v1.x+1: Update all examples to use new naming
- v1.x+2: Update documentation to recommend new naming
- v2.0: Remove deprecated methods (breaking change)

**Option B: Direct Migration (Faster but riskier)**
- v2.0: Rename all methods, publish migration guide
- Provide automated migration script

### Phase 3: Implementation (v2.0)

#### Core Methods to Rename

**stochastic_benchmark.py:**
```python
# Current → New
initAll() → init_all()
run_Bootstrap() → run_bootstrap()
run_Interpolate() → run_interpolate()
run_Stats() → run_stats()
get_TrainingResults_recipe() → get_training_results_recipe()
get_TrainingStats_recipe() → get_training_stats_recipe()
```

**Other modules:**
```python
# bootstrap.py
Bootstrap() → bootstrap()
BootstrapSingle() → bootstrap_single()
Bootstrap_reduce_mem() → bootstrap_reduce_mem()

# stats.py
Stats() → stats()
StatsSingle() → stats_single()

# interpolate.py
Interpolate() → interpolate()
Interpolate_reduce_mem() → interpolate_reduce_mem()
```

#### Files to Update

1. **Source Code:**
   - [ ] `src/stochastic_benchmark.py`
   - [ ] `src/bootstrap.py`
   - [ ] `src/stats.py`
   - [ ] `src/interpolate.py`
   - [ ] `src/training.py`
   - [ ] `src/random_exploration.py`
   - [ ] `src/sequential_exploration.py`
   - [ ] Other modules as needed

2. **Tests:**
   - [ ] `tests/test_bootstrap.py`
   - [ ] `tests/test_stats.py`
   - [ ] `tests/test_stochastic_benchmark.py` (if exists)
   - [ ] All integration tests
   - [ ] All other test files

3. **Examples:**
   - [ ] `examples/QAOA_iterative/`
   - [ ] `examples/QAOA_multipleSplits/`
   - [ ] `examples/wishart_n_50_alpha_0.5/`
   - [ ] Any other example directories

4. **Documentation:**
   - [ ] `README.md`
   - [ ] `TESTING.md`
   - [ ] API documentation
   - [ ] Tutorial notebooks
   - [ ] Inline code comments

### Phase 4: Migration Support

#### Provide Migration Tools

**Option 1: Automated Script**
```python
# migrate_naming.py
"""
Script to automatically update code from v1.x to v2.0 naming conventions.

Usage:
    python migrate_naming.py path/to/your/code
"""

import re
import sys
from pathlib import Path

REPLACEMENTS = {
    r'\.initAll\(': '.init_all(',
    r'\.run_Bootstrap\(': '.run_bootstrap(',
    r'\.run_Interpolate\(': '.run_interpolate(',
    r'\.run_Stats\(': '.run_stats(',
    # Add all other replacements
}

def migrate_file(filepath):
    """Update a single file to v2.0 naming."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    for old, new in REPLACEMENTS.items():
        content = re.sub(old, new, content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Implementation continues...
```

**Option 2: Migration Guide**
Create a comprehensive guide in `MIGRATION_v1_to_v2.md`:
```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes

### Method Naming

All public API methods now use snake_case instead of PascalCase.

| Old (v1.x) | New (v2.0) |
|------------|------------|
| `sb.initAll()` | `sb.init_all()` |
| `sb.run_Bootstrap()` | `sb.run_bootstrap()` |
| ... | ... |

## Quick Migration Steps

1. Find and replace in your codebase
2. Run tests
3. Update any custom extensions
```

### Phase 5: Release Strategy

**v1.x (Current):**
- Continue bug fixes and critical updates
- No new features with old naming
- Document deprecation plan

**v1.x+1 (Deprecation Start):**
- Add all new snake_case methods
- Old methods remain but emit DeprecationWarning
- Update all examples to use new naming
- Release migration guide

**v1.x+2 (Final Warning):**
- Increase warning visibility
- Update all documentation
- Provide automated migration script
- Set firm v2.0 release date

**v2.0 (Breaking Release):**
- Remove all deprecated methods
- Only snake_case API available
- Clean, PEP 8 compliant codebase

## Success Criteria

- [ ] All public methods use snake_case
- [ ] All tests pass with new naming
- [ ] All examples updated and working
- [ ] Documentation completely updated
- [ ] Migration guide published
- [ ] Migration script tested on example codebases
- [ ] No reduction in test coverage
- [ ] User feedback incorporated

## Timeline Estimate

- **Phase 1 (Inventory)**: 1-2 weeks
- **Phase 2 (Deprecation)**: 3-6 months across multiple releases
- **Phase 3 (Implementation)**: 2-3 weeks
- **Phase 4 (Migration Support)**: 2 weeks
- **Phase 5 (Release)**: Following standard release cycle

**Total**: ~6-9 months from start to v2.0 release

## Risks and Mitigations

### Risk: Breaking User Code
**Mitigation**: Long deprecation period with clear warnings

### Risk: Incomplete Migration
**Mitigation**: Automated testing, comprehensive checklists

### Risk: Documentation Drift
**Mitigation**: Update docs simultaneously with code

### Risk: User Confusion
**Mitigation**: Clear communication, migration tools

## Alternative Approaches

### Keep Current Naming
**Pros**: No breaking changes, existing code continues to work
**Cons**: Continues to violate PEP 8, may confuse Python developers

### Gradual Module-by-Module
**Pros**: Smaller, more manageable changes
**Cons**: Inconsistent API during transition, longer timeline

### Aliases Forever
**Pros**: Backward compatible
**Cons**: Maintains technical debt, larger codebase

## Decision

**Recommended Approach**: Option A (Dual API with Deprecation Period)

This provides the best balance of:
- User experience (time to adapt)
- Code quality (eventual PEP 8 compliance)
- Project maintainability (clean v2.0 codebase)

## Next Steps

1. Get stakeholder buy-in on this plan
2. Create GitHub issue to track this work
3. Begin Phase 1 (Inventory) when ready to start v2.0 planning
4. Communicate plan to users via:
   - GitHub issue/discussion
   - Release notes
   - Documentation
   - Email/blog if applicable

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 8 - Naming Conventions](https://peps.python.org/pep-0008/#naming-conventions)
- [Semantic Versioning](https://semver.org/)

---

**Status**: Proposed  
**Created**: 2025-10-21  
**Author**: GitHub Copilot  
**Version**: Draft 1.0
