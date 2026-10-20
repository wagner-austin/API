# The system interpreter commit-msg, post-commit and pre-push launch
# published_maketools.py with, sourced by each so it is chosen once.
#
# Spelled as MCPs' .husky/run-at-head.sh and scripts/make/shell.mk spell it:
# `python` on Windows, where the launcher owns `python3`, and `python3` on
# Debian-family Linux, where `python` is not provided. Until MCPs board task
# 6c0664bf each hook named `python`, which a Linux session host such as
# ziggurat does not have. Git for Windows' sh is bash, which sets OSTYPE to
# msys; a POSIX sh elsewhere may leave it unset.
case "${OSTYPE:-}" in
  msys* | cygwin*) HOOK_PYTHON=python ;;
  *) HOOK_PYTHON=python3 ;;
esac
