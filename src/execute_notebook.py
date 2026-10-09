"""Execute and validate the companion; export HTML and plot previews."""
from pathlib import Path
import base64
import os
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def execute():
    path = Path('notebooks/economic-neighbors.ipynb')
    notebook = nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook,timeout=180,kernel_name='python3',
                            resources={'metadata':{'path':str(Path.cwd())}})
    client.execute()
    nbformat.validate(notebook)
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
    nbformat.write(notebook,path)
    preview_dir = Path('outputs/notebook-preview'); preview_dir.mkdir(parents=True,exist_ok=True)
    html,_ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    (preview_dir/'economic-neighbors.html').write_text(html)
    count = 0
    for cell in notebook.cells:
        for output in cell.get('outputs',[]):
            png = output.get('data',{}).get('image/png')
            if png:
                count += 1
                (preview_dir/f'figure-{count}.png').write_bytes(base64.b64decode(png))
    assert count==5, f'Expected five meaningful figures, got {count}'
    print(f'Executed {sum(c.cell_type=="code" for c in notebook.cells)} code cells; {count} figures; HTML preview saved')


if __name__=='__main__': execute()
