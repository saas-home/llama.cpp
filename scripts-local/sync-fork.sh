#!/bin/bash
set -euo pipefail

# ====================== USAGE ==============================
# ./sync-fork.sh                  → full sync: sync master + rebase current + push
# ./sync-fork.sh --branch <name>  → specifies branch to sync (defaults to current)
# ./sync-fork.sh --force           → force-push after rebase (required after history change)
# ./sync-fork.sh --dry-run        → show what would be synced (no changes)
# ./sync-fork.sh --no-master       → skip syncing the local master branch
# ============================================================

DRY_RUN=false
FORCE_PUSH=false
SYNC_MASTER=true
BRANCH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true ; shift ;;
        --force) FORCE_PUSH=true ; shift ;;
        --no-master) SYNC_MASTER=false ; shift ;;
        --branch) BRANCH="$2" ; shift 2 ;;
        *) echo "Unknown parameter: $1" ; exit 1 ;;
    esac
done

# ====================== CONFIGURATION ======================
FORK_REMOTE="origin"
UPSTREAM_REMOTE="upstream"
UPSTREAM_BRANCH="master"

# ====================== HELPER FUNCTIONS ====================
print_header() {
    echo ""
    echo "=========================================================="
    echo "  $1"
    echo "=========================================================="
}

print_step() {
    echo "> $1"
}

print_success() {
    echo "[OK] $1"
}

print_error() {
    echo "[ERROR] $1"
}

print_info() {
    echo "[INFO] $1"
}

# ====================== MAIN WORKFLOW ======================

# 1. Auto-detect llama.cpp directory
LLAMA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LLAMA_DIR"
print_success "Using llama.cpp directory: $LLAMA_DIR"

# 2. Verify remotes
print_step "Verifying git remotes..."
if ! git remote | grep -q "^$FORK_REMOTE$"; then
    print_error "Fork remote '$FORK_REMOTE' not found"
    exit 1
fi
if ! git remote | grep -q "^$UPSTREAM_REMOTE$"; then
    print_error "Upstream remote '$UPSTREAM_REMOTE' not found"
    exit 1
fi

# 3. Detect current branch
CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")
if [[ -z "$BRANCH" ]]; then
    BRANCH="$CURRENT_BRANCH"
fi

if [[ "$BRANCH" == "detached" ]]; then
    print_error "Cannot sync in a detached HEAD state."
    exit 1
fi
print_success "Target branch: $BRANCH"

if [[ "$DRY_RUN" == true ]]; then
    print_header "DRY RUN MODE"
    print_step "Would perform the following:"
    echo "  1. git push $FORK_REMOTE $BRANCH (to save work)"
    echo "  2. git checkout $UPSTREAM_BRANCH && git merge $UPSTREAM_REMOTE/$UPSTREAM_BRANCH && git push $FORK_REMOTE $UPSTREAM_BRANCH"
    echo "  3. git checkout $BRANCH && git rebase $UPSTREAM_BRANCH"
    echo "  4. git push $([[ "$FORCE_PUSH" == true ]] && echo "-f ") $FORK_REMOTE $BRANCH"
    exit 0
fi

# 4. Save current work (only if not on master)
if [[ "$BRANCH" != "$UPSTREAM_BRANCH" ]]; then
    print_header "Step 0: Saving current progress on $BRANCH"
    print_step "Running: git push $FORK_REMOTE $BRANCH"
    # We use a standard push here to ensure we don't overwrite remote by accident before rebase
    if git push "$FORK_REMOTE" "$BRANCH"; then
        print_success "Progress saved to $FORK_REMOTE/$BRANCH"
    else
        print_info "Push failed or branch is already up to date. Continuing..."
    fi
fi

# 5. Sync Master Branch
if [[ "$SYNC_MASTER" == true ]]; then
    print_header "Step 1: Synchronizing local $UPSTREAM_BRANCH with $UPSTREAM_REMOTE"
    
    # Check for uncommitted changes before switching
    if ! git diff-index --quiet HEAD --; then
        print_error "Uncommitted changes detected. Stash or commit them before running sync."
        exit 1
    fi

    if [[ "$BRANCH" != "$UPSTREAM_BRANCH" ]]; then
        print_step "Switching to $UPSTREAM_BRANCH"
        git checkout "$UPSTREAM_BRANCH"
    fi
    
    print_step "Fetching and merging from $UPSTREAM_REMOTE/$UPSTREAM_BRANCH"
    git fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH"
    if git merge "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"; then
        print_success "Master is now up to date with upstream"
    else
        print_error "Failed to merge upstream/$UPSTREAM_BRANCH into master"
        exit 1
    fi
    
    print_step "Pushing updated $UPSTREAM_BRANCH to $FORK_REMOTE"
    git push "$FORK_REMOTE" "$UPSTREAM_BRANCH"
    
    if [[ "$BRANCH" != "$UPSTREAM_BRANCH" ]]; then
        print_step "Returning to $BRANCH"
        git checkout "$BRANCH"
    fi
fi

# 6. Rebase current branch (if not on master)
if [[ "$BRANCH" != "$UPSTREAM_BRANCH" ]]; then
    print_header "Step 2: Rebasing $BRANCH on $UPSTREAM_BRANCH"
    print_step "Running: git rebase $UPSTREAM_BRANCH"
    if git rebase "$UPSTREAM_BRANCH"; then
        print_success "Rebase completed successfully"
    else
        print_error "Rebase conflict detected"
        echo "To resolve:"
        echo "  1. Fix conflicts"
        echo "  2. git add <files>"
        echo "  3. git rebase --continue"
        exit 1
    fi

    # 7. Push rebased branch
    print_header "Step 3: Pushing $BRANCH to $FORK_REMOTE"
    if [[ "$FORCE_PUSH" == true ]]; then
        print_step "Running: git push -f $FORK_REMOTE $BRANCH"
        if git push -f "$FORK_REMOTE" "$BRANCH"; then
            print_success "Successfully force-pushed $BRANCH"
        else
            print_error "Force-push failed"
            exit 1
        fi
    else
        print_step "Running: git push $FORK_REMOTE $BRANCH"
        if git push "$FORK_REMOTE" "$BRANCH"; then
            print_success "Successfully pushed $BRANCH"
        else
            print_error "Push failed. You likely need to use --force after a rebase."
            exit 1
        fi
    fi
fi

print_header "Synchronization Complete"
print_info "Local branch : $(git symbolic-ref --short HEAD)"
print_info "Status       : Up to date with upstream/master"
echo ""
