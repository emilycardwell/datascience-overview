# Terminal Setup

- add new directory/subdir/files
```
mkdir my-project
cd my-project
mkdir subdir1
mkdir subdir2
mkdir subdir3
touch subdir3/file1.py
touch subdir3/__init__.py
code .
```

- add global environmental variables
```
code ~/.zshrc
export PYTHONPATH='/Users/.../my-project'
export PYTHONPATH='/Users/…/new_functions:$PYTHONPATH’
```

- check python path and root_dir path list for package imports
```
import os; os.environ['PYTHONPATH']
import sys; sys.path
```

- restart terminal
```
exec zsh
```

- Github CLI
```
# send local repo to gh
git init -b main
git add . && git commit -m ‘initial commit’
gh repo create
- push an existing repo
- yes

# get local repo from remote
gh repo clone git@github.com:emilycardwell/repo.git
```

- VS code - light integrated development environment
    - extenstions: python, pylance, 4-space indent, subline text keymap
- terminal: (zsh shell)
    - pwd (find directory)
    - ls (directory files)
        - ls -l (list view of directory files w/ info)
    - mkdir test (make directory - “test”)
    - cd test (change directory to test)
    - cd .. (takes you one step higher up parent directory)
    - cd ../.. (go back two levels)
    - cd ~ (or just cd) (back to home directory)
        - .. for every parent directory
    - touch test.txt (create file)
    - rm test.txt (delete file)
    - rm -rf test (removes test directory: must include recursive and force)
    - clear (scrolls down so it makes terminal clean (doesn’t delete things)
    - mv test/test.txt test.txt (moves from one location to another [ first, second ])
    - mv text.txt emily.txt (renames file)
    - code (opens vscode window)
    - code emily.txt (opens file in vscode)
    - . (the current directory (pwd)
    - code . (opens terminal opening code in vscode)
    - echo (print)
    - echo $PATH (prints directory path)
        - pwd (print the path of the current directory)
    - echo $PATH | tr : \\ (prints directory but with new line instead of : between each)
    - **debugging command not found:**
        - type -a code (shows path of code: $PATH)
- pyenv (installs and manages multiple python versions)
    - pyenv versions (shows all downloaded python versions)
    - pyenv activate  <environment name>
- virtualenv: environment with correct versions and types of packages that’s needed for a specific project
- pip list (lists all installed packages)
    - pip list (show versions of packages)
    - pip list | grep pandas (shows only versions of pandas)
- git and github - versional saves
    - git close < on github code / close / ssh >
    - git remote -v
    - git add index.html
    - git commit -m
    - git push origin master (<remote> <branch>)
    - git stash
    - git status (if you have updated to current version)
        - git diff <a specific file/folder>
        - inspect the detailed changes of the modified files
    - make (tool to run commands against a dependency tree defined in a Makefile)
    - **commit your changes to a git repo**
        - git status
        - git add <modfile1>
        - git add <modfile2>
        - git commit —message ‘your message here’
        - git push origin master
