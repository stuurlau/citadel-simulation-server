# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['runcitadel']

package_data = \
{'': ['*']}

install_requires = \
['jsonschema>=4.17.3,<5.0.0', 'websockets>=10.4,<11.0']

setup_kwargs = {
    'name': 'runcitadel',
    'version': '2.0.0',
    'description': 'A package for developing and running citadel simulations.',
    'long_description': '# runcitadel\n\nA package for developing and running citadel simulations.\n\n## Installation\n\n```bash\n$ pip install runcitadel\n```\n\n## Usage\n\n`runcitadel` can be used to connect a local simulation to a citadel session to perform\ninteractive computation on graphs. For usage examples, see the examples notebook in /docs/.\n\n## Contributing\n\nInterested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.\n\n## License\n\n`runcitadel` was created by Miles van der Lely. It is licensed under the terms of the MIT license.\n\n## Credits\n\n`runcitadel` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).\n\nDeveloped in the Visualisation Lab at the University of Amsterdam (UvA).\n',
    'author': 'Miles van der Lely',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
