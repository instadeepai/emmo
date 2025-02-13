# Contributing

In order to contribute to `emmo` you will need developer access to this repo.

## How to contribute

1. Create an issue, cf [How to create an issue](CONTRIBUTING.md#how-to-create-an-issue) if you are
   not familiar with the workflow
1. Assign yourself to the issue
1. Click on "Create merge request" in the issue page: it will create a branch with the pattern
   `{issue_id}-{issue_title_slug}` and the corresponding merge request, to which you should assign
   yourself.
1. Locally, fetch and checkout the branch:
   ```bash
   git fetch
   git checkout {branch} # use autocomplete by writing {issue_id}- and then press TAB
   ```
1. Implement your changes, commit and push. :warning: your commit messages must follow our
   [git conventions](CONTRIBUTING.md#git-conventions)
1. Once you are done, put your merge request as ready (i.e. remove the "Draft" from the merge
   request title): in the top right corner, click on the button on the right of "Code" and click on
   "Mark as ready"
1. Assign a reviewer
1. Integrate the feedbacks if any:
   1. The author of the threads (most often the reviewer) on an MR is in charge of marking them as
      resolved in GitLab UI.
   1. When integrating the suggestions from the reviewer, mark the MR as "Draft". Once done, mark it
      as ready and re-request a review by clicking on the round arrow in the "Reviewer" section.
1. Once your merge request is approved:
   - Update your branch with the latest version of `main`:
   ```bash
   git fetch && git rebase origin/main && git push -f
   ```
   - Clean your commits if necessary: you should keep one commit per task (e.g. if in your branch
     you did a refactoring and a new feature you should have 2 commits)
   - Merge :tada:

## How to create an issue

The GitLab issues are used to track all our tasks (e.g. feature, bug fix, ...).

1. Go to the [issues list](https://gitlab.com/instadeep/emmo/-/issues)
1. Click on "New issue button" in the top right corner
1. Fill the title of the issue
1. Choose a template, the following templates available are:
   - Bug
   - Build
   - CI
   - Data
   - Documentation
   - Experiment
   - Feature
   - Literature
   - Performance
   - Refactor
   - Test
1. Fill the sections in the template selected
1. Click on "Create issue" in the bottom of the page

## Git conventions

### Commit message format

- The section relies on the
  [Contributing to Angular - Commit Message Guidelines](https://github.com/angular/angular/blob/master/CONTRIBUTING.md#commit)
- It provides conventions to write commits messages based on the
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/)
- It aims to :
  - Get a well-structured and easily understandable git history
  - Generate changelogs easily for each release since we can use scripts that parse the commit
    messages
- The commit messages must have the following structure :

```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

- `<type>` section :
  - It is mandatory
  - It must be one of the following :
    - `build`: Changes to our deployment configuration (e.g. docker, requirements, pre-commit
      configuration)
    - `ci` : Changes to our CI configuration files and scripts
    - `docs` : Documentation changes
    - `feat` : A new feature
    - `fix` : A bug fix
    - `perf` : A code change that improves performance
    - `refactor` : A code change that neither fixes a bug nor adds a feature
    - `style` : Changes that do not affect the meaning of the code (white-space, formatting, missing
      semi-colons, etc)
    - `test` : Adding missing tests or correcting existing tests
- `<scope>` section :
  - It is optional
  - It describes the module affected by the changes
- `<subject>` section :
  - It is mandatory
  - It contains a succinct description of the change
  - Few recommendations about the subject :
    - use the imperative, present tense: "change" not "changed" nor "changes"
    - don't capitalize the first letter
    - no dot (.) at the end
- `<body>` section :
  - It is optional
  - It is an extension of the `<subject>` section used to add a longer description about the changes
    if relevant
- `<footer>` section :
  - It is optional
  - It can contain information about breaking changes and is also the place to reference GitLab
    issues, that this commit closes or is related to.

Example commit message:

```
feat: allow provided config object to extend other configs

BREAKING CHANGE: `extends` key in config file is now used for extending other config files
```

- You can add the commit message template to the git configuration by running :

```bash
git config commit.template $PWD/.gitmessage
```

### Merge request rules

- The commit history shall be as atomic as possible: one commit per task.
- For very simple and trivial modifications (e.g. typo correction, very light refactor), it is not
  necessary to create a dedicated merge request, but simply possible to integrate them into one of
  your current MR.

  **Note:** The author or reviewer may prefer a dedicated merge request if the modification is
  already (or becomes) too large and requires a proper review.

### Git checklists

#### For authors 🧑‍💻 (before marking your MR as ready)

- Check that **the commit history is clean**, i.e. explicit and comprehensive; and that the
  modifications of your commits are consistent and atomic.
- Check that all your commits **respect the present guidelines**, notably
  [commit message format](CONTRIBUTING.md#commit-message-format).

#### For reviewers 🕵️ (before approval)

- Check changes: do not hesitate to give feedback for any questions or concerns. **Feedback is good
  for everyone, authors and reviewers!**
- Check that all commits **respect the present guidelines**. In particular, check that the history
  does not contain superfluous commits (like experiment commit if it is a merge request into
  `main`); request to have these commits removed if necessary.
- Make your suggestions/comments in one batch instead of one by one. For that, you need to click on
  the "Start a review" button on your first comment and then on the "Add to review" button on the
  next ones. Once completed, you can click on "Finish review".

## Pre-commit hooks

To ensure code quality we are using [pre-commit hooks](https://pre-commit.com/), make sure you have
installed it! (cf. [Setup - Step-by-step](getting_started/install.md#step-by-step))

## Style Guide

Here are guidelines regarding some aspects that couldn't be checked by the `pre-commit`:

- For docstring:
  - Follow the
    [Google style guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
  - Do not specify typing.
- If a function is dealing with tensors or numpy arrays (as arguments or returned value), put the
  shape in the docstring description and use `?` for batch size, e.g. `(?, seq_len)`.
- Use `f-string` to format variables in string instead of `%` or `format`.
- Use list comprehension instead of `map`/`filter`.
- Use `pathlib.Path` to deal with local paths and `cloudpathlib.CloudPath` to deal with remote
  paths. If a variable can be either local or remote use `cloudpathlib.AnyPath`. For the typing, you
  should use `pathlib.Path | cloudpathlib.CloudPath`.

  ??? Example

      ```python
      from __future__ import annotations

      from pathlib import Path

      import pandas as pd
      from cloudpath import AnyPath
      from cloudpath import CloudPath

      def load_dataset(file_path: Path | CloudPath) -> pd.DataFrame:
        ...

      my_file_path = AnyPath("..")
      ```

- Add blank line before the `return` statement in a function, except:

  - when the function body is only the `return` statement.

    ??? Example

        ```python
        def my_func(param: int) -> int:
          """Super docstring."""
          return 2 * param
        ```

  - if the `return` statement is after a logical statement.

    ??? Example

        ```python
        def my_func(param: int) -> int:
          """Super docstring."""
          if param % 2 == 0:
            return param + 4

          return 2 * param
        ```

- Do not add blank line after `for`/`if`/`elif`/`else`/`while` statements.
- When dealing with tensors or numpy arrays, put the shape as comment in the line before and use `?`
  for batch size.

  ??? Example

      ```python
      peptide = self.inputs["peptide"]
      peptide_encoder = self._build_input_encoder(peptide)

      # (?, peptide_len)
      peptide_input = peptide.build_input_layer()

      # (?, peptide_len, d_model)
      peptide_encoded = peptide_encoder(peptide_input)
      ```

- Use comment only if the code is not self explanatory.
- Use blank lines in functions to separate logical blocks.
- Do not add a blank line between `try`/`except`/`else`/`finally` statements.

  ??? Example

      ```python
      try:
        result = x // y
        print(f"Your answer is : {result}")
      except ZeroDivisionError:
        print("You are dividing by zero")
      ```

- If you need a container for membership test, e.g.
  `if "test" in {"test", "another_test"}: print("Found!")`, use a `set` as
  [advised by `pylint`](https://pylint.pycqa.org/en/latest/user_guide/messages/refactor/use-set-for-membership.html).
- If you need a container for iteration, e.g. `for element in ("test", "TEST", 4)`, prefer using a
  `tuple` over a `list`, a `frozenset`, or a `set`, since it is
  [advised by `pylint`](https://pylint.pycqa.org/en/latest/user_guide/messages/convention/use-sequence-for-iteration.html)
  to use `list`/`tuple`/`range` for iteration and since `tuple` is faster to be created (see. the
  following benchmark). If you consider iterating over consecutive numbers though, use `range` as it
  is more convenient/readable.

  ??? Tip

      ```python
      def from_list():
          return ["test", "TEST", 4]

      def from_tuple():
          return ("test", "TEST", 4)

      def from_set():
          return {"test", "TEST", 4}

      def from_frozenset():
          return frozenset(("test", "TEST", 4))

      %timeit from_list() # 84.9 ns ± 1.44 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
      %timeit from_tuple() # 56.4 ns ± 0.687 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
      %timeit from_set() # 106 ns ± 3.99 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
      %timeit from_frozenset() # 177 ns ± 1.21 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
      ```

- For the typing of a generator which only yield values (i.e. no definition of `send` or `return`),
  you should use `collections.abc.Iterator[YieldType]` instead of
  `collections.abc.Generator[YieldType, None, None]`.

## Naming conventions

- For MHC classes we should use: MHC1/mhc1 and MHC2/mhc2
