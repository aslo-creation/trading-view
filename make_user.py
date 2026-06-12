"""
make_user.py — provision a user for auth_config.yaml.

Usage:
    python make_user.py alice trader
Prompts for the password without echoing it, prints a YAML block containing
ONLY the bcrypt hash. Plaintext is never written anywhere.
"""

from __future__ import annotations

import getpass
import sys

from config.security import hash_password

ROLES = {"admin", "trader", "viewer"}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in ROLES:
        print(f"Usage: python make_user.py <username> <{'|'.join(sorted(ROLES))}>")
        return 1
    username, role = sys.argv[1], sys.argv[2]
    if not username.isalnum() or len(username) > 32:
        print("Username must be alphanumeric, max 32 chars.")
        return 1

    pw1 = getpass.getpass("Password (min 10 chars): ")
    pw2 = getpass.getpass("Confirm: ")
    if pw1 != pw2:
        print("Passwords do not match.")
        return 1

    hashed = hash_password(pw1)
    print("\nPaste under credentials.usernames in config/auth_config.yaml:\n")
    print(f"    {username}:")
    print(f"      name: {username}")
    print(f"      role: {role}")
    print(f"      password: {hashed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
