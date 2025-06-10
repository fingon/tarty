# Python code conventions #

- Assume most recent Python

  - e.g. don't use Dict or Optional from typing

- Try to minimize number of external dependencies in production code (test dependencies are fine)

- If updating code, update also e.g. corresponding READIME

- @staticmethod is a code smell. Just use plain functions

- Do not make obvious comments which can be inferred from the nearby function call

- Use module level logger, and call it explicitly always instead of adding utility functions for creating logs

- Do not use format strings in logging

## Test code conventions ##

- Write pytest unit tests

- Make the code unit testable using fakes, instead of writing mocks for every function

- Do not comment the test code at all
