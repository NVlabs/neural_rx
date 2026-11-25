# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import base64
import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOKS = sorted((REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
EXECUTED_NOTEBOOKS = [
    path for path in NOTEBOOKS if path.stem != "site_specific_neural_receivers"
]


@pytest.mark.parametrize("path", EXECUTED_NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_is_saved_executed_with_figures(path: Path) -> None:
    notebook = json.loads(path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is not None for cell in code_cells
               if ''.join(cell['source']).strip())
    assert not any(output['output_type'] == 'error'
                   for cell in code_cells for output in cell.get('outputs', []))
    for cell in code_cells:
        if 'plt.subplots(' in ''.join(cell['source']):
            images = [output.get('data', {}).get('image/png') for output in cell.get('outputs', [])]
            assert any(images), f'{path.name}: plotting cell has no embedded PNG'
            for image in filter(None, images):
                assert base64.b64decode(''.join(image)).startswith(b'\x89PNG\r\n\x1a\n')
    assert any(cell.get('outputs') for cell in code_cells)


def test_generalized_training_is_opt_in():
    name = 'jumpstart_tutorial'
    notebook = json.loads((REPOSITORY_ROOT / 'notebooks' / f'{name}.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert 'RUN_FULL_TRAINING = False' in source
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and '.backward()' in source:
            assert 'if RUN_FULL_TRAINING' in source


@pytest.mark.parametrize("path", EXECUTED_NOTEBOOKS, ids=lambda path: path.stem)
@pytest.mark.notebook
def test_notebook_executes(path: Path, tmp_path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    notebook_client = pytest.importorskip("nbclient")
    notebook = nbformat.read(path, as_version=4)
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    kernel = tmp_path / 'kernels' / 'neural-rx-test'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Neural RX test interpreter', 'language': 'python',
    }))
    manager = KernelManager(kernel_name='neural-rx-test', kernel_spec_manager=KernelSpecManager(
        kernel_dirs=[str(kernel.parent)],
    ))
    notebook_client.NotebookClient(
        notebook,
        timeout=None,
        km=manager,
        resources={"metadata": {"path": str(REPOSITORY_ROOT)}},
    ).execute(cleanup_kc=True)
