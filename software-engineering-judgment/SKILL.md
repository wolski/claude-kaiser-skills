---
name: software-engineering-judgment
description: This skill should be used whenever the user asks to "review this architecture", "refactor this module", "simplify this API", "where should this code live", "should this be a new module/class/protocol", "remove duplication", or "improve maintainability" in any programming language. Apply it during substantial code design and review even when the request is phrased as an implementation task. It provides cross-language judgment about cohesion, ownership, abstraction, polymorphism, optionality, module boundaries, public API size, and evidence-driven refactoring. Combine it with the relevant language-specific skill for syntax and ecosystem conventions.
---

# Software Engineering Judgment

## Purpose

Design code around responsibilities that already exist in the domain. Prefer the
smallest structure that expresses current behavior clearly, preserves invariants,
and leaves a straightforward path for proven future needs. Treat every module,
interface, protocol, factory, configuration option, and public symbol as a cost that
must earn its place.

Apply these principles across Python, R, Java, TypeScript, C++, Rust, and other
languages. Combine them with language-specific guidance rather than replacing
idiomatic language features.

## Start with the Ownership Question

Before adding code, identify the natural owner of the behavior:

1. Locate the data and invariants required by the operation.
2. Locate the subsystem that already implements the underlying domain behavior.
3. Ask which existing type or module can perform the operation without reaching
   through unrelated layers.
4. Add behavior to that owner when doing so keeps dependencies coherent.
5. Create a new owner only when no existing owner fits and the new responsibility
   can be named independently.

Use one diagnostic question before creating a file:

> Does this module remove a dependency or own a distinct concept, or does it merely
> add another hop?

Reject a new module when its main job is importing one existing module, selecting a
branch, and forwarding arguments. Such a module increases navigation cost without
creating a useful boundary.

## Let Boundaries Earn Their Cost

Create a module or component when it establishes at least one concrete boundary:

- own an independently testable domain invariant;
- isolate an external dependency or volatile integration;
- enforce a dependency direction between subsystems;
- provide a stable concept used by multiple real consumers;
- contain enough related behavior that the existing owner would lose cohesion;
- define a lifecycle, resource, or state boundary that must be controlled explicitly.

Before extraction, write one sentence in the form: "This component owns ___." Reject
the extraction when the blank can only be filled with "calling", "routing",
"wrapping", or "preparing for future implementations". Prefer a private helper inside
the existing owner when the behavior is useful but not independently meaningful.

Reassess a proposed boundary when it causes several files to change for one simple
behavior, requires callers to understand an intermediate representation, or introduces
configuration solely to reconstruct the old default. These are signs that the boundary
tracks implementation steps rather than a domain concept.

## Prefer Cohesion over Orchestration

Keep behavior beside the state and rules it needs. Avoid central functions that
inspect concrete variants and then reproduce knowledge already held by those
variants.

When several concrete input types support the same domain operation, prefer a
same-named method or an existing language dispatch mechanism:

```text
positioned = queue_input.position_queue()
```

Prefer this over a procedural dispatcher:

```text
position_queue(queue_input):
    if queue_input is VialInput:
        ...
    else if queue_input is PlateInput:
        ...
```

Call this correction **cohesive polymorphism**: place the operation on the concrete
types, preserve one conceptual API, and let ordinary dispatch choose the behavior.
Keep shared mechanics in the subsystem that already owns them.

Do not force methods onto data types when the operation is genuinely external,
changes independently of those types, or would introduce an inverted dependency.
In those cases, prefer the language's established visitor, multimethod, pattern
matching, or service pattern. Make that choice because of dependency direction and
change patterns, not because another abstraction might someday be useful.

## Require Evidence before Abstracting

Treat imagined reuse as speculation. Introduce an abstraction only after concrete
pressure demonstrates what varies and what stays stable.

- Keep one implementation concrete.
- At the second similar implementation, compare the cases and record the variation.
- At the third real case, consider extracting the common contract.
- Extract only the stable intersection, not every imaginable option.
- Keep the new abstraction internal until an external consumer requires it.

Do not add a strategy, protocol, callback, factory, plugin point, or generic type for
a single implementation. A parameter that exists only to choose the sole default is
not flexibility; it is deferred design work pushed onto every caller.

Name this failure **speculative generality** or **premature abstraction**. Common
signals include:

- an interface with one implementation;
- a strategy parameter with no production alternative;
- a factory that always constructs one concrete class;
- a wrapper module with no independent invariant;
- comments describing hypothetical consumers rather than current requirements;
- input parsing or validation generalized past the shapes the rest of the system can
  produce or consume — for example a regex admitting multi-letter tokens when every
  consumer assumes a single letter — where a check against existing domain data (a
  known set of valid values) would be simpler and stricter;
- a larger test surface without additional user-visible behavior.

## Keep APIs Honest and Explicit

Make required state required. Avoid absence values that secretly mean "construct a
default", "select the normal behavior", "load from somewhere else", or "decide
later".

Prefer:

```text
generate(positioned_queue)
```

over:

```text
generate(queue, config = absent, assigner = absent)
```

Treat optional parameters and nullable fields as design decisions, not conveniences.
Allow absence only when absence is a genuine domain value. Otherwise choose one of
these forms:

- require the value;
- provide a concrete default value or default implementation;
- expose separate operations with distinct names;
- represent lifecycle states with distinct types;
- resolve defaults at the construction boundary and persist the resolved value.

This principle applies to `None`, `NULL`, `nil`, `null`, missing options, empty
configuration objects, and equivalent sentinels. Do not convert a useful hard
failure into implicit fallback behavior.

Keep public surfaces minimal. Avoid exposing internal helpers, intermediate models,
or customization hooks until a caller needs them. A narrow API permits internal
change; a speculative API creates compatibility obligations.

## Remove Duplication at the Right Level

Distinguish duplicated text from duplicated knowledge. Two similar lines are not
automatically a problem. Remove duplication when both copies encode the same rule and
must change together.

For adjacent branches that produce the same kind of intermediate value, let the
branch choose that value and run common processing once afterward:

```text
if vial input:
    plate_queue = assign_positions(input)
else if plate input:
    plate_queue = input.queue

plate_queue = validate_positions(plate_queue)
```

Do not create a new helper or module merely to reduce line count. Prefer a local,
obvious common tail over an abstraction with a vague name. Optimize for one-pass
readability: a reviewer should understand ownership and control flow without jumping
between files repeatedly.

## Keep Side Effects at Commit Boundaries

Separate pure transformation from persistence, network calls, logging artifacts, and
external publication. Let domain operations return explicit results. Persist only at
the point where the user or system commits to an action such as saving, downloading,
uploading, or executing.

This separation improves preview behavior, testing, reproducibility, and failure
handling. It also prevents a method named `build`, `preview`, or `validate` from
silently changing external state.

## Refactor toward the Root Cause

When code becomes awkward, identify the structural cause before adding a wrapper.
Use the following sequence:

1. Describe the awkwardness with a concrete example.
2. Identify which responsibility is misplaced or duplicated.
3. Move that responsibility to its natural owner.
4. Remove obsolete branches, parameters, modules, and tests.
5. Update call sites to use the resulting direct API.
6. Verify public behavior before introducing further flexibility.

Avoid **refactor ping-pong**: adding a layer, simplifying the code inside it, moving
the operation elsewhere, and finally deleting the layer. Prevent it by testing the
ownership question before extraction.

## Let Code Document Itself

Let the code state what it does; reserve comments and docstrings for what the code
cannot express — the *why* behind a non-obvious choice, an invariant a reader must not
violate, a genuine gotcha, and a public symbol's contract (arguments, return, raised
errors). Never restate the code, and never narrate a change ("no longer holds X", "now
derives Y", "split moved elsewhere"): that documents a diff rather than the program, so
it belongs in the commit message and goes stale in the file.

## Review Checklist

Before accepting a design or refactor, verify:

- **Responsibility:** Can every changed module state what it owns in one sentence?
- **Placement:** Does behavior live beside the state and invariants it uses?
- **Dispatch:** Does a type switch duplicate behavior that concrete types could own?
- **Evidence:** Does every new abstraction have multiple real uses or a required
  dependency boundary?
- **Optionality:** Does every absent value represent legitimate domain absence?
- **API:** Is each new public symbol required by an actual caller?
- **Navigation:** Does the change reduce conceptual hops rather than add them?
- **Duplication:** Was duplicated knowledge removed without hiding simple control
  flow?
- **Effects:** Are persistence and external mutations confined to explicit commit
  points?
- **Deletion:** Were superseded helpers, branches, imports, tests, and documentation
  removed?
- **Comments:** Does each comment carry a why, an invariant, or a contract, rather than
  restate the code or narrate what changed?
- **Verification:** Do tests cover observable behavior and important invariants rather
  than preserving an obsolete implementation shape?

## Communicate Design Decisions

Describe architectural changes using four concise elements:

1. Name the observed problem.
2. Name the natural owner of the behavior.
3. Explain why the chosen boundary is smaller and more cohesive.
4. State the evidence and verification supporting the change.

When correcting an overdesigned solution, name both sides clearly:

- **Antipattern:** speculative generality combined with a pass-through orchestration
  layer and procedural type dispatch.
- **Correction:** cohesive polymorphism, evidence-driven abstraction, and colocation
  with the existing domain subsystem.

Prefer direct explanations over design-pattern vocabulary when the vocabulary does
not improve the decision. The goal is readable software, not demonstrating familiarity
with patterns.
