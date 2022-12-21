#!/bin/bash

git commit -a -F- <<EOF
"$1"
`git status | grep -e '^\(*\|  > \)'`
EOF
