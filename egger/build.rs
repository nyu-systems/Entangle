// build.rs
// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::Path;

fn main() {
    let template_dir = "template/special";
    let dst_dir = "src/special";

    for p in vec!["special_lemma.rs", "special_model.rs"] {
        let dst_path = Path::new(dst_dir).join(p);
        if !Path::exists(&dst_path) {
            let template_path = Path::new(template_dir).join(p);
            println!(
                "cargo::warning=MESSAGE: Special implementation not found, copying {} to {}",
                template_path.display(),
                dst_path.display()
            );
            std::fs::copy(template_path, dst_path).unwrap();
        }
    }
}
