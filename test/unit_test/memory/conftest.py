#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
conftest.py — shims for running memory unit tests without the full
RAGFlow dependency tree (numpy, elasticsearch, strenum, etc.).

The project requires Python >=3.12, but these unit tests are designed
to be runnable in isolation.  We mock out the heavy ``common.doc_store``
module so that ``NativeMemoryStore`` can be imported without pulling in
numpy, elasticsearch-dsl, etc.
"""
import sys
import types

# Only install shims if the real modules are NOT available.
# This keeps the tests honest when run inside the full venv.

def _install_shim(module_path, attrs=None):
    """Register a lightweight stub module if the real one can't be imported."""
    try:
        __import__(module_path)
        return  # real module available — nothing to do
    except Exception:
        pass

    parts = module_path.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            sys.modules[partial] = types.ModuleType(partial)

    mod = sys.modules[module_path]
    for name, obj in (attrs or {}).items():
        setattr(mod, name, obj)


# ── OrderByExpr shim ──
class _OrderByExpr:
    def __init__(self):
        self.fields = []
    def asc(self, field):
        self.fields.append((field, 0))
        return self
    def desc(self, field):
        self.fields.append((field, 1))
        return self

_install_shim("common.doc_store.doc_store_base", {
    "OrderByExpr": _OrderByExpr,
    "MatchExpr": object,
    "MatchTextExpr": type("MatchTextExpr", (), {}),
    "MatchDenseExpr": type("MatchDenseExpr", (), {}),
    "MatchSparseExpr": type("MatchSparseExpr", (), {}),
    "MatchTensorExpr": type("MatchTensorExpr", (), {}),
    "FusionExpr": type("FusionExpr", (), {}),
    "DocStoreConnection": type("DocStoreConnection", (), {}),
    "SparseVector": type("SparseVector", (), {}),
    "VEC": object,
    "DEFAULT_MATCH_VECTOR_TOPN": 10,
    "DEFAULT_MATCH_SPARSE_TOPN": 10,
})
