# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import re

import paddle
from datetime import date


def kernel_filter(all_kernel_list):
    valid_kernel_list = []
    # phi_ops_all = paddle.fluid.core.get_all_op_names("phi")
    for opk in all_kernel_list:
        op_name = opk
        if type(opk) is tuple:
            # print("opk:", opk)
            op_name = opk[0]

        if op_name.endswith("_sparse_grad"):
            continue
        if op_name.endswith("_grad_grad"):
            continue
        if op_name.endswith("_double_grad"):
            continue
        if op_name.endswith("_triple_grad"):
            continue
        if op_name.endswith("_sparse"):
            continue
        if op_name.startswith("sparse_"):
            continue
        if op_name.startswith("graph_send_"):
            continue
        # if op_name.startswith("fused_"):
        #    continue
        if op_name.startswith("fft_"):
            continue
        if op_name.startswith("feed"):
            continue
        if op_name.startswith("fetch"):
            continue
        # if op_name.startswith("c_"):
        #    continue
        if op_name.startswith("run_program"):
            continue
        if op_name.startswith("partial_"):
            continue
        if op_name.startswith("fake_"):
            continue
        if op_name.startswith("quantize_"):
            continue
        if op_name.startswith("dequantize_"):
            continue
        # if "inplace" in op_name:
        #    continue
        valid_kernel_list.append(opk)
    return valid_kernel_list


def get_data_type(data_str):
    m = re.search(r"data_type\[([a-z0-9\_\:\<\>]+)\]\;\s", data_str)
    # print(m)
    if m:
        found = m.group(1)
        if "::paddle::platform::" in found:
            r = found.split("::paddle::platform::")
            # print("split:", r[1])
            return r[1]
        else:
            return found

    return None


def get_kernel_by_place(target_place):
    op_support_list = []
    kernel_dict = paddle.core._get_all_register_op_kernels("phi")
    for op_name in kernel_dict:
        kernel_list = kernel_dict[op_name]
        dtype = []
        find = False

        # kenrel on only cpu
        only_cpu = False

        for item in kernel_list:
            print(op_name, item)
            # print(f"op_name = {op_name}, item = {item}", get_data_type(item))
            # if get_data_type(item) is None:
            #    print(op_name, item)
            if item.find(target_place) != -1 or only_cpu:
                dtype.append(get_data_type(item))
                find = True

        if (find or only_cpu) and op_name not in op_support_list:
            op_support_list.append((op_name, dtype))

    return op_support_list


def get_target_place():
    # get target place and output file name
    # place = "dcu" if paddle.is_compiled_with_rocm() else sys.argv[1]
    # output_file = f"{place}_ops.csv"
    target_place = f"place[Place({sys.argv[1]}:0)]"
    print("target_place:", target_place)
    if sys.argv[1] == "cpu":
        target_place = f"place[Place({sys.argv[1]})]"
    return target_place


def dump_kernels_with_dtype(op_dtype_list, file_name):
    with open(file_name, "w") as f:
        for item in op_dtype_list:
            f.write("%s," % item[0])
            for dtype in item[1]:
                f.write("%s:" % dtype)
            f.write("\n")


if __name__ == "__main__":
    target_place = get_target_place()
    print("place:", target_place)

    # for debug - phi ops filter
    phi_ops_all = paddle.core.get_all_op_names("phi")
    phi_ops_all.sort()
    with open("phi_ops_all.csv", "w") as f:
        for item in phi_ops_all:
            f.write("%s\n" % item)
    phi_ops_cln = kernel_filter(phi_ops_all)
    phi_ops_cln.sort()
    with open("phi_ops_cln.csv", "w") as f:
        for item in phi_ops_cln:
            f.write("%s\n" % item)

    # for debug - dev ops filer
    dev_ops_all = get_kernel_by_place(target_place)
    dev_ops_all.sort()
    dump_kernels_with_dtype(dev_ops_all, "dev_ops_all.csv")

    dev_ops_cln = kernel_filter(dev_ops_all)
    dev_ops_cln.sort()
    dump_kernels_with_dtype(dev_ops_cln, "dev_ops_cln.csv")

    kernel_list = []
    for op_dtype in dev_ops_cln:
        if op_dtype[0] in phi_ops_cln:
            kernel_list.append(op_dtype)

    place = "dcu" if paddle.is_compiled_with_rocm() else sys.argv[1]
    today = date.today().strftime("%Y-%m-%d")
    output_file = f"{place}_ops_" + today + ".csv"

    dump_kernels_with_dtype(kernel_list, output_file)
