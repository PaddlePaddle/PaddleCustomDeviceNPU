# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import warnings
from paddle_sdaa.sdaa_ext import *  # noqa
import paddle
from paddle.optimizer import AdamW
from paddle.base import framework
import numpy as np
import paddle.profiler as profiler
from paddle.base.layer_helper import LayerHelper
from ..utils import *  # noqa
from .device_map import *  # noqa
from ..storage import *  # noqa
from .distributed_optimizer import DistributeOptimizer


class DistributeAdamW(AdamW, DistributeOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_param = None
        self.rank_flat_param = None
        self._rank_param_group = None
        self.clipped_param = {}
        self.flat_accum = {}
        self.flatten_params = []
        self._already_flat_acc = set()
        self.amp_o2 = False  # if traversed
        # all groups
        self.groups = []
        self.t_block = framework.default_main_program().global_block()
        # my group
        self.group = None
        self.total_rank = None
        self.helper = LayerHelper(self.__class__.__name__)
        self.HIGH_PERFORMANCE_CONV = int(os.environ.get("HIGH_PERFORMANCE_CONV", "0"))
        self.need_append_all_param = True
        if self.HIGH_PERFORMANCE_CONV:
            warnings.warn(
                "DistributeOptimizer Now Not Support HIGH_PERFORMANCE_CONV , Distrbuteoptimizer would work as origin optimizer"
            )
            return
        if (
            self._grad_clip is None
            or isinstance(self._grad_clip, paddle.nn.ClipGradByValue)
            or isinstance(self._grad_clip, paddle.nn.ClipGradByNorm)
        ):
            self.need_append_all_param = False
        if paddle.in_dynamic_mode():
            self.re_distribution()
            if self.group is not None and not isinstance(self._parameter_list[0], dict):
                self.re_flatten()
            self._create_accumulators(
                self.t_block,
                [
                    param
                    for param in self.flatten_params
                    if "eager_tmp" not in param.name
                ],
            )
            self._flatten_accumulators()
            warnings.warn(
                "DistributeOptimizer would add all trainable param to accumulators"
            )
            warnings.warn(
                "DistributeOptimizer Now Not Support HIGH_PERFORMANCE_CONV , Only Support Param^dtype = Fp32 and Param^shape = NCHW,Please Check Amp Level and Param^shape"
            )

    def _allgather_accumulators(self):
        if self.HIGH_PERFORMANCE_CONV:
            return
        for k, _ in self.flat_accum.items():
            self.group.process_group.all_gather_partial_on_calc_stream(
                self.flat_accum[k], self.flat_accum[k], self.group.world_size, self.rank
            )

    def _need_flatten(self):
        if self.HIGH_PERFORMANCE_CONV:
            return
        if not self.flat_accum:
            return True

    def _flatten_accumulators(self):
        if self.HIGH_PERFORMANCE_CONV:
            return
        for k, _ in self._accumulators.items():
            # only flatten moment
            if "beta" in k:
                continue
            flatacc_list = []
            total_num = 0

            for param in self.flatten_params:
                if "eager_tmp" in param.name:
                    continue
                flatacc_list.append(self._accumulators[k][param.name])
                self._already_flat_acc.add(param.name)
                # padding align to 128B, now we assume param is float32, this should change
                numel = (
                    (np.prod(self._accumulators[k][param.name].shape) + 31) // 32 * 32
                )
                total_num += numel

            # for all_gather, every rank allocate_flatten_tensor should align 128B, assume param is float32,is 128/4 =32
            align_size = 32 * self.total_rank
            if total_num % align_size != 0:
                total_num_ = (total_num + align_size - 1) // align_size * align_size
                # print(total_num_, total_num, flush=True)
                flatacc_list.append(
                    paddle.empty([total_num_ - total_num], dtype=paddle.float32)
                )
                total_num = total_num_

            self.flat_accum[k] = paddle.empty(shape=[total_num], dtype=paddle.float32)
            paddle._legacy_C_ops.coalesce_tensor(
                flatacc_list,
                flatacc_list,
                self.flat_accum[k],
                "copy_data",
                True,
                "use_align",
                True,
                "align_size",
                128,
                "dtype",
                flatacc_list[0].dtype,
            )

    def _update_beta(self, name, param):
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype)
        target_param = self._master_weights[param.name] if find_master else param
        target_name = target_param.name
        if (
            name not in self._accumulators
            or target_name not in self._accumulators[name]
        ):
            raise Exception(
                "Accumulator {} does not exist for parameter {}".format(
                    name, target_name
                )
            )
        if "beta1" in name:
            beta = (
                self._beta1
                if not isinstance(self._beta1, paddle.static.Variable)
                else self._beta1.item(0)
            )
        elif "beta2" in name:
            beta = (
                self._beta2
                if not isinstance(self._beta2, paddle.static.Variable)
                else self._beta2.item(0)
            )
        else:
            raise Exception(
                "Accumulator ony have beta1_pow and beta2_pow ,but accet {}".format(
                    name
                )
            )
        self._accumulators[name][target_name].scale_(beta)

    def _append_optimize_op(self, block, param_and_grad):

        if (
            self.HIGH_PERFORMANCE_CONV
            or self.group is None
            or isinstance(param_and_grad, dict)
        ):
            return super()._append_optimize_op(block, param_and_grad)
        if self._rank_param_group[self.rank].get(param_and_grad[0].name) is None:
            return
        param, grad = param_and_grad
        should_clip, start, end = self._rank_param_group[self.rank][param.name]
        # Whether we should do weight decay for the parameter.
        with_decay = True
        if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(
            param.name
        ):
            with_decay = False

        moment1 = self._get_accumulator_master(self._moment1_acc_str, param_and_grad[0])
        moment2 = self._get_accumulator_master(self._moment2_acc_str, param_and_grad[0])
        beta1_pow_acc = self._get_accumulator_master(
            self._beta1_pow_acc_str, param_and_grad[0]
        )
        beta2_pow_acc = self._get_accumulator_master(
            self._beta2_pow_acc_str, param_and_grad[0]
        )
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name] if find_master else None
        )
        lr = self._create_param_lr(param_and_grad)

        lr_ratio_ = 1.0 if self._lr_ratio is None else self._lr_ratio(param_and_grad[0])

        _beta1 = (
            self._beta1
            if not isinstance(self._beta1, paddle.static.Variable)
            else self._beta1.item(0)
        )
        _beta2 = (
            self._beta2
            if not isinstance(self._beta2, paddle.static.Variable)
            else self._beta2.item(0)
        )

        if (
            should_clip
            and tensor_storage_format(param) == "NCHW"
            and tensor_storage_format(grad) == "NCHW"
        ):
            if self.clipped_param.get(param.name) is None:
                param_ = self.helper.create_global_variable(
                    name=f"{param.name}_clip_{self.rank}_param",
                    persistable=False,
                    dtype=param.dtype,
                    shape=[end - start],
                    belong_to_optimizer=True,
                )

                grad_ = self.helper.create_global_variable(
                    name=f"{param.name}_clip_{self.rank}_grad",
                    persistable=False,
                    dtype=grad.dtype,
                    shape=[end - start],
                    belong_to_optimizer=True,
                )

                moment1_ = self.helper.create_global_variable(
                    name=f"{param.name}_clip_{self.rank}_moment1",
                    persistable=False,
                    dtype=moment1.dtype,
                    shape=[end - start],
                    belong_to_optimizer=True,
                )

                moment2_ = self.helper.create_global_variable(
                    name=f"{param.name}_clip_{self.rank}_moment2",
                    persistable=False,
                    dtype=moment2.dtype,
                    shape=[end - start],
                    belong_to_optimizer=True,
                )
                temp = paddle.empty([])
                shape = param.shape
                temp.get_tensor()._share_data_with(param_and_grad[0].get_tensor())
                param_ = temp.reshape_([-1])._slice(start, end)
                temp.get_tensor()._share_data_with(param_and_grad[1].get_tensor())
                grad_ = temp.reshape_([-1])._slice(start, end)
                moment1_ = moment1.reshape_([-1])._slice(start, end)
                moment2_ = moment2.reshape_([-1])._slice(start, end)
                moment1.reshape_(shape)
                moment2.reshape_(shape)
                param_.stop_gradient = True
                grad_.stop_gradient = True
                moment1_.stop_gradient = True
                moment2_.stop_gradient = True
                self.clipped_param[param.name] = (param_, moment1_, moment2_)
            else:
                param_, moment1_, moment2_ = self.clipped_param[param.name]
                temp = paddle.empty([])
                temp.get_tensor()._share_data_with(param_and_grad[1].get_tensor())
                grad_ = temp.reshape_([-1])._slice(start, end)
        else:
            param_ = param_and_grad[0]
            grad_ = param_and_grad[1]
            moment1_ = moment1
            moment2_ = moment2

        _, _, _, _, _, _ = paddle._C_ops.adamw_(
            param_,
            grad_,
            lr,
            moment1_,
            moment2_,
            beta1_pow_acc,
            beta2_pow_acc,
            master_weight,
            None,
            _beta1,
            _beta2,
            self._epsilon,
            lr_ratio_,
            self._weight_decay,
            with_decay,
            self._lazy_mode,
            1000,
            find_master,
            False,
        )
        return None

    def step(self):
        record_event = profiler.RecordEvent("optimizer_step")
        record_event.begin()
        found_inf = self._get_auxiliary_var("found_inf")
        if self.HIGH_PERFORMANCE_CONV:
            super().step()
        elif (
            self.group is not None
            and not isinstance(self._parameter_list[0], dict)
            and self._rank_param_group is not None
        ):
            params_grads = []
            for param in self._parameter_list:
                if self.amp_o2 is False:
                    assert (
                        param.dtype == paddle.float32
                    ), "distributed optimizer only support amp level1"
                if param.stop_gradient or param._grad_ivar() is None:
                    continue
                if (
                    self._rank_param_group[self.rank].get(param.name) is None
                    and not found_inf
                ):
                    self._update_beta(self._beta2_pow_acc_str, param)
                    self._update_beta(self._beta1_pow_acc_str, param)
                grad_var = param._grad_ivar()
                if paddle.in_dynamic_mode():
                    if (
                        hasattr(grad_var, "is_selected_rows")
                        and grad_var.is_selected_rows()
                        and self.regularization is not None
                    ):
                        raise RuntimeError(
                            "AdamW don't support weight_decay with sparse parameters, please set it to None."
                        )
                else:
                    if (
                        hasattr(grad_var, "_is_sparse")
                        and grad_var._is_sparse()
                        and self.regularization is not None
                    ):
                        raise RuntimeError(
                            "AdamW don't support weight_decay with sparse parameters, please set it to None."
                        )
                if (
                    self.need_append_all_param
                    or self._rank_param_group[self.rank].get(param.name) is not None
                ):
                    params_grads.append((param, grad_var))
            self.amp_o2 = True
            optimize_ops = self._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads
            )
            # Because some param is clip , so need all_gather
            self.group.process_group.all_gather_partial_on_calc_stream(
                self.flat_param, self.flat_param, self.group.world_size, self.rank
            )
            if self._need_flatten():
                self._flatten_accumulators()
            else:
                pass
        else:
            super().step()
        record_event.end()

    def minimize(self, loss, startup_program=None, parameters=None, no_grad_set=None):
        found_inf = self._get_auxiliary_var("found_inf")
        if self.HIGH_PERFORMANCE_CONV:
            return super().minimize(loss, startup_program, parameters, no_grad_set)
        elif (
            self.group is not None
            and not isinstance(self._parameter_list[0], dict)
            and self._rank_param_group is not None
        ):
            assert isinstance(
                loss, paddle.static.Variable
            ), "The loss should be an Tensor."

            parameter_list = parameters if parameters else self._parameter_list

            params_grads = self.backward(
                loss,
                startup_program=startup_program,
                parameters=parameter_list,
                no_grad_set=no_grad_set,
            )

            new_params_grads = []
            for param, grad in params_grads:
                if param.stop_gradient or grad is None:
                    continue
                if (
                    self._rank_param_group[self.rank].get(param.name) is None
                    and not found_inf
                ):
                    self._update_beta(self._beta2_pow_acc_str, param)
                    self._update_beta(self._beta1_pow_acc_str, param)
                if (
                    self.need_append_all_param
                    or self._rank_param_group[self.rank].get(param.name) is not None
                ):
                    new_params_grads.append((param, grad))
            optimize_ops = self._apply_optimize(
                loss, startup_program=startup_program, params_grads=new_params_grads
            )
            self.group.process_group.all_gather_partial_on_calc_stream(
                self.flat_param, self.flat_param, self.group.world_size, self.rank
            )
            if self._need_flatten():
                self._flatten_accumulators()
            else:
                pass
            return optimize_ops, params_grads
        else:
            return super().minimize(loss, startup_program, parameters, no_grad_set)
