# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""GIGO supported operators."""
from tvm.relay import transform
import tvm.ir

@tvm.ir.register_op_attr("nn.conv2d", "target.gigo")
def conv2d(attrs, args):
    return True

@tvm.ir.register_op_attr("qnn.conv2d", "target.gigo")
def qnn_conv2d(attrs, args):
    return True

def partition_for_gigo(mod):
    """Perform graph partition to offload supported operators to gigo hw

    Parameters
    ----------
    mod : Module
        The module to run passes on.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    seq = tvm.transform.Sequential([transform.AnnotateTarget('gigo'),
                                    transform.MergeCompilerRegions(),
                                    transform.PartitionGraph()])
    return seq(mod)
