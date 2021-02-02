#! /bin/bash
bumpversion --current-version $(grep version setup.py | sed -E 's/.*version.*=.*"([[:digit:]]+\.[[:digit:]]\.[[:digit:]]+)".*/\1/') patch setup.py