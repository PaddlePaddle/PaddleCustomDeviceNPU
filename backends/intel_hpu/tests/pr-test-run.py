#!/usr/bin/python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import os
import argparse
import fnmatch

realwd = os.path.dirname(os.path.realpath(__file__))
pdpd_pcd_unit_test_path = os.environ.get(
    "PDPD_PCD_UNITTEST_DIR", "/PaddleCustomDevice/backends/intel_hpu/tests/unittests"
)

os.environ.setdefault("GAUDI2_CI", "1")


def case_command(command, test_case=None, test_suite=None):
    output = ""
    output += f"Command: {command}\n"
    proc = subprocess.Popen(
        command, bufsize=0, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        line = proc.stdout.readline().decode()
        while len(line) > 0:
            print(f"{line[:-1]}")  # line include '\n' at last
            output += line
            line = proc.stdout.readline().decode()
    except Exception as e:
        if test_case:
            test_case.setFail("Command abnormally")
        print(e)
    finally:
        proc.communicate()
    if proc.returncode != 0 and test_case:
        test_case.setFail(f"Command return code:{proc.returncode}")
    if test_case:
        test_case.AddOutput(output)
    if test_suite:
        ts.addCase(test_case)

    return proc.returncode, output


def run_unit_test(test_suite, test_case, cmd_env=""):
    testcase_name = test_case.get("testcase_name", "")
    testcase_path = test_case.get("testcase_path", "")

    run_cmd = f"python {testcase_path}"
    _env = os.environ.copy()
    for opt in cmd_env.split():
        _env.setdefault(opt.split("=")[0], opt.split("=")[1])
    print(f"RUN shell CMD: {cmd_env} {run_cmd}")

    testcase = None
    if test_suite:
        testcase = jTestCase(testcase_name)
    else:
        testcase = None

    ret, output = case_command(run_cmd, testcase, test_suite)


cmd_args = {}
parser = argparse.ArgumentParser(
    description="help scriopt for run the test",
    add_help=True,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--test_path",
    type=str,
    help="path of PaddleCustomDevice unit tests folder",
    default=pdpd_pcd_unit_test_path,
)
parser.add_argument("--platform", type=str, help="platform name")
parser.add_argument("--junit", type=str, help="junit result file")
parser.add_argument("--k", type=str, help="designative test case name to run")

cmd_args.update(vars(parser.parse_args()))

if cmd_args["junit"]:
    libpath = os.path.dirname(os.path.dirname(realwd))
    if os.path.exists(f"{libpath}/junitxml.py"):
        import sys

        sys.path.append(libpath)
    from junitxml import jTestSuite, jTestCase

test_platform = "Gaudi2"
if cmd_args["platform"]:
    test_platform = cmd_args["platform"]

if os.path.exists(cmd_args["test_path"]) is False:
    print(
        "test path not exist, please check for parameter: test_path / environment PDPD_PCD_UNITTEST_DIR"
    )
    exit(2)

test_case_dict_lst = []

pdpd_pcd_unittest_path = cmd_args["test_path"]
for file in os.listdir(os.path.dirname(pdpd_pcd_unittest_path)):
    if fnmatch.fnmatch(file, "*.py"):
        absfile = os.path.join(os.path.dirname(pdpd_pcd_unittest_path), file)
        test_case_dict = {}
        test_case_dict["testcase_name"] = file
        test_case_dict["testcase_path"] = absfile
        test_case_dict_lst.append(test_case_dict)

ts = None
if cmd_args["junit"]:
    ts = jTestSuite("PaddleCustomDevice/UnitTests")
    ts.setPlatform(test_platform)

match_test_case_name = ""
if cmd_args["k"]:
    match_test_case_name = cmd_args["k"]
    match_test_case_name = match_test_case_name.strip()

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

for test_case in test_case_dict_lst:
    testcase_name = test_case.get("testcase_name", "")
    if match_test_case_name and match_test_case_name not in testcase_name:
        continue
    run_unit_test(ts, test_case)

if cmd_args["junit"]:
    with open(cmd_args["junit"], "w+") as f:
        f.write(ts.toString())
    print(ts.toString(True))
