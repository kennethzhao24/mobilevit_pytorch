import os
from setuptools import find_packages, setup

VERSION = 0.1


def do_setup(package_data):
    setup(
        name="mobilevit_pytorch",
        version=VERSION,
        setup_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "torch",
            "tqdm",
        ],
        packages=find_packages(
            exclude=[
                "config_files",
                "config_files.*"
            ]
        ),
        package_data=package_data,
        test_suite="tests",
        zip_safe=False,
    )


def get_files(path, relative_to="."):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    package_data = {
        "lib": (
            get_files(os.path.join("lib", "config"))
        )
    }
    do_setup(package_data)
