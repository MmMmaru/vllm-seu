# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Participant evaluation wrapper, importable as an installed vLLM subpackage.

After installing this vLLM fork (editable or wheel), the DNDX participant
wrapper is available at ``vllm.eval.evaluation_wrapper`` so that benchmark
code can import it from any working directory:

    from vllm.eval.evaluation_wrapper import VLMModel, GenerationConfig
"""