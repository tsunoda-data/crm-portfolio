import json
import sys
import os

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cells = []
    current_cell_type = 'code'
    current_source = []

    def flush_cell():
        if not current_source:
            return
        # Remove trailing empty lines
        while current_source and current_source[-1].strip() == '':
            current_source.pop()
        
        if not current_source:
            return

        cell = {
            "cell_type": current_cell_type,
            "metadata": {},
            "source": current_source
        }
        if current_cell_type == 'code':
            cell["execution_count"] = null = None
            cell["outputs"] = []
            
        cells.append(cell)

    for line in lines:
        if line.startswith('# %%'):
            flush_cell()
            current_source = []
            if '[markdown]' in line:
                current_cell_type = 'markdown'
            else:
                current_cell_type = 'code'
        else:
            if current_cell_type == 'markdown':
                # Remove the leading '# ' for markdown cells if present
                if line.startswith('# '):
                    current_source.append(line[2:])
                elif line.startswith('#\n'):
                    current_source.append('\n')
                else:
                    current_source.append(line)
            else:
                current_source.append(line)
                
    flush_cell()

    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Created {ipynb_file}")

if __name__ == '__main__':
    py_to_ipynb(sys.argv[1], sys.argv[2])
