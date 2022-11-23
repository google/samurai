# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import sys


def run_method(basefolder, method, method_args):
    method_args_str = " ".join(method_args)
    command = f"PYTHONPATH={basefolder}/external/{method} conda run -n {method} python -m external_run.{method}.run_script {method_args_str}"
    full_command = 'bash -c "' + command + '"'
    print(full_command)
    subprocess.run(full_command, stderr=sys.stderr, stdout=sys.stdout, shell=True)


if __name__ == "__main__":
    import argparse

    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    methods = [
        e
        for e in os.listdir("external_run")
        if os.path.isdir(os.path.join("external_run", e))
    ]
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", required=True, choices=methods)

    args, method_args = parser.parse_known_args()

    run_method(parent, args.method, method_args)
