#! /bin/bash
old_version=$(grep version setup.py | sed -E 's/.*version.*=.*"([[:digit:]]+\.[[:digit:]]\.[[:digit:]]+)".*/\1/')
bumpversion --current-version $old_version patch setup.py
new_version=$(grep version setup.py | sed -E 's/.*version.*=.*"([[:digit:]]+\.[[:digit:]]\.[[:digit:]]+)".*/\1/')
echo "::set-output name=some_output::$new_version"