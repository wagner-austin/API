# The system interpreter commit-msg, post-commit and pre-push launch
# published_maketools.py with, sourced by each so it is chosen once.
#
# Spelled as MCPs' .husky/run-at-head.sh and scripts/make/shell.mk spell it:
# `python3`, since Debian-family Linux provides no `python`. Until MCPs board
# task 6c0664bf each hook named `python`, which a Linux session host such as
# ziggurat does not have.
HOOK_PYTHON=python3
