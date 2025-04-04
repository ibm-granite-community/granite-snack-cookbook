# Prompt Declaration Language

Two similar notebooks are provided to illustrate [PDL](https://github.com/IBM/prompt-declaration-language):

* The `Prompt_Declaration_Language.ipynb` notebook uses our convenient Jupyter extension for writing PDL programs.
  * Outside the notebook the PDL interpreter can be used as a CLI on YAML files directly.
* The `Prompt_Declaration_Language_python.ipynb` notebook has the same content, but it uses the Python API instead of the notebook extension. This is useful for several reasons:
  * It illustrates PDL Python SDK, for more information see [here](https://github.com/IBM/prompt-declaration-language).
  * We can run our automated tests on this notebook, because these tests convert the notebooks to pure Python scripts, which doesn't work for the `Prompt_Declaration_Language.ipynb` notebook.
  * You can use Python string formatting (e.g., f"Hello {name}") when constructing the PDL program.

> **Pro Tip:** PDL uses the variable expansion convention `${...}`. If you use Python string formatting, use `${{...}}` so Python doesn't attempt to interpolate those strings!
