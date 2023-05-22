from setuptools import setup, find_packages, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")
    
def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]

setup(
  name = 'salesforce-codetf',
  version = "0.0.1",
  py_modules = ['codetf'],
  description = 'CodeTF: A Transformer-based Library for Code Intelligence',
  author = 'Nghi D. Q. Bui',
  long_description=open("README.md", "r", encoding="utf-8").read(),
  long_description_content_type="text/markdown",
  keywords="AI4Code, Code Intelligence, Generative AI, Deep Learning, Library, PyTorch, HuggingFace",
  license="3-Clause BSD",
  url = 'https://github.com/Salesforce/CodeTF',
  packages=find_namespace_packages(include="codetf.*"),
  install_requires=fetch_requirements("requirements.txt"),
  include_package_data=True,
  zip_safe=False,
  python_requires=">=3.8.0",
)