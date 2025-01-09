# How to run all unit tests for PR testing
# setup the PYTHONPATH env
export PYTHONPATH=/workspace/pdpd_ci/PaddleCustomDevice/python/:/workspace/pdpd_ci/PaddleCustomDevice/Paddle/test/legacy_test/

# common cmdline
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/

# to run all test cases under the --test_path folder with xml output
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2

# to run special test cases
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2 --k test_abs_op
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2 --k test_abs_op.py
