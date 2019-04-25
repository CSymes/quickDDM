#!/bin/bash

# Just a wrapper around Python's inbuilt unittest discovery engine
# Runs all tests in the `tests` folder
# Run with `-v` to get verbose test logging
# @author: Cary

python -m unittest discover tests $@
