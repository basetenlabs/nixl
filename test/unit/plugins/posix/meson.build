# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Using globally defined has_posix_plugin from the root meson.build
if has_posix_plugin
    # Get Abseil dependencies
    absl_log_dep = dependency('absl_log', required: true)

    nixl_posix_app = executable('nixl_posix_test', 'nixl_posix_test.cpp',
                                dependencies: [nixl_dep, nixl_infra, absl_log_dep],
                                include_directories: [nixl_inc_dirs, utils_inc_dirs],
                                install: true)

    # Register the test with the test suite
    test('posix_plugin_test', nixl_posix_app)

    # VRAM bounce buffer tests (requires CUDA GPU)
    cuda_rt_dep = dependency('cudart', required: false)
    if not cuda_rt_dep.found()
        cuda_rt_dep = meson.get_compiler('cpp').find_library('cudart',
            dirs: get_option('cudapath_lib') != '' ? [get_option('cudapath_lib')] : ['/usr/local/cuda/lib64'],
            required: false)
    endif

    if cuda_rt_dep.found()
        cuda_inc = include_directories(
            get_option('cudapath_inc') != '' ? get_option('cudapath_inc') : '/usr/local/cuda/include')

        nixl_posix_vram_app = executable('nixl_posix_vram_test',
            'nixl_posix_vram_test.cpp',
            dependencies: [nixl_dep, nixl_infra, absl_log_dep, cuda_rt_dep],
            include_directories: [nixl_inc_dirs, utils_inc_dirs, cuda_inc],
            install: true)

        test('posix_vram_test', nixl_posix_vram_app,
             env: ['CUFILE_ENV_PATH_JSON=/tmp/cufile_test.json'])
    endif
endif
