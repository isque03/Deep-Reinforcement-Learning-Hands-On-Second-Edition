# Instructions for updating branch on another machine

# Option 1: Rename local branch and update tracking (Recommended)
# If you're currently on the gan-implementation branch:

git fetch origin
git branch -m gan-implementation torch-2.6
git branch --set-upstream-to=origin/torch-2.6 torch-2.6
git pull

# Option 2: Checkout the new branch directly
# If you want to switch to the new branch name:

git fetch origin
git checkout torch-2.6
git branch -D gan-implementation  # Delete old local branch name

# Option 3: If you have uncommitted changes
# First stash or commit your changes, then:

git fetch origin
git checkout -b torch-2.6 origin/torch-2.6
git branch -D gan-implementation
