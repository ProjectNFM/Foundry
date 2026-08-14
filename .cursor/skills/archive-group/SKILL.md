---
name: archive-group
description: >-
  Synthesize and archive a batch of completed inbox experiments into a named
  group directory with a synthesis README. Use when the user says "archive",
  "group experiments", "synthesize results", "clean up inbox", or when multiple
  completed experiments in inbox/ address a common research thread.
disable-model-invocation: true
---

# Archive Group

Synthesize completed experiments and move them from inbox into a thematic
group directory.

## Workflow

### Step 1: Scan Inbox for Completed Experiments

Read all files in `experiments/inbox/` and identify those with
`Status: Completed`.

- Parse parent/child links to identify clusters of related experiments
- Group by shared tags, common parents, or overlapping research questions
- Present findings to the user

If fewer than 2 completed experiments exist in inbox, inform the user and
suggest waiting for more results before archiving.

### Step 2: Propose Grouping (Interactive)

Present the proposed cluster to the user and ask for confirmation:

```
question: |
  I found N completed experiments that form a coherent thread:
  
  • <filename> — <title>
  • <filename> — <title>
  • ...
  
  Proposed group: "<group-name>" — <one-sentence thesis>
  
  Does this grouping look right?
options:
  - "Yes, archive these as a group"
  - "Add/remove experiments from this group (I'll specify)"
  - "Split into two separate groups"
  - "Don't archive yet — more experiments coming for this thread"
```

If the user wants to adjust, iterate until the grouping is confirmed.

### Step 3: Name the Group (Interactive)

Determine the next available group number by scanning existing
`experiments/<NN>-*/` directories.

```
question: "Name for the archive directory?"
options:
  - "<NN>-<suggested-slug> (Recommended)"
  - "<NN>-<alternative-1>"
  - "<NN>-<alternative-2>"
  - "Let me provide a custom name"
```

### Step 4: Synthesize Findings

Read the Conclusions and Results sections from each experiment in the group.
Draft a group `README.md` with:

```markdown
# <Group Title>

**Experiments:** <count>
**Date range:** YYYY-MM-DD to YYYY-MM-DD
**Contributors:** <initials>

## Overarching Question

<One sentence: what research question ties these experiments together?>

## Summary of Findings

<3–5 paragraph narrative synthesizing results across all experiments.
Draw connections between individual findings. Highlight how each experiment
informed the next.>

## Key Takeaways

- <Design decision or insight #1 — with reference to supporting experiment>
- <Design decision or insight #2>
- ...

## Experiment Index

| # | Experiment | Hypothesis Verdict | Key Metric |
|---|-----------|-------------------|------------|
| 1 | [<title>](./<filename>) | Confirmed/Refuted | <metric: value> |
| ... | ... | ... | ... |

## Open Questions

- <What remains unanswered that future work should address?>
```

### Step 5: Review Synthesis (Interactive)

Present the draft README to the user for review:

```
question: "Is this synthesis accurate and complete?"
options:
  - "Yes, looks good — proceed with archiving"
  - "Needs edits — I'll mark what to change"
  - "Missing a key insight (I'll add it)"
```

Iterate until approved.

### Step 6: Execute File Operations

1. Create directory: `experiments/<NN>-<slug>/`
2. Write `experiments/<NN>-<slug>/README.md`
3. Move experiment files: `git mv experiments/inbox/<file> experiments/<NN>-<slug>/`
4. Update relative links in ALL moved files:
   - Parent/child experiment links
   - Analysis script references
   - Figure paths
5. Update references in any remaining inbox experiments or other archived
   groups that point to the moved files.

### Step 7: Verify Integrity

After moving files:
- Check that no broken relative links exist in the moved files
- Check that no external files have dangling references to old inbox paths
- Confirm the inbox only contains non-grouped experiments
- Run `git status` to show the user what changed

## Key Principles

- **User approves everything.** Never move files without explicit confirmation
  of the grouping, naming, and synthesis content.
- **Atomic commits.** All file moves + link fixes should happen in a single
  commit to avoid intermediate broken states.
- **Group numbers are for sort order only.** They're assigned at archive time
  by incrementing from the highest existing group number. They carry no
  semantic meaning.
- **Synthesis is not summary.** The README should draw NEW conclusions by
  connecting findings across experiments, not just list individual results.
- **Preserve git history.** Always use `git mv` (not copy+delete) so that
  `git log --follow` traces file history.
