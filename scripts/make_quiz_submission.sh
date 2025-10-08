#!/usr/bin/env bash
set -e
python -m ebayner.predict --config configs/default.yaml --quiz --to-submission
