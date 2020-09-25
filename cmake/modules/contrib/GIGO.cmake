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

if(USE_GIGO)
    file(GLOB GIGO_RELAY_CONTRIB_SRC src/relay/backend/contrib/gigo/*.cc)
    list(APPEND COMPILER_SRCS ${GIGO_RELAY_CONTRIB_SRC})

    file(GLOB GIGO_RUNTIME_CONTRIB_SRC
      CONFIGURE_DEPENDS src/runtime/contrib/gigo/*.cc)
    list(APPEND RUNTIME_SRCS ${GIGO_RUNTIME_CONTRIB_SRC})

    message(STATUS "Build with GIGO runtime")
endif()
