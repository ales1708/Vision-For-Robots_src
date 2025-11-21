from setuptools import find_packages, setup

package_name = "det_loc"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "detection_subscriber = det_loc.detection_subscriber:main",
            "localization_subscriber = det_loc.localization_subscriber:main",
            "rectify_images = det_loc.rect_images:main",
        ],
    },
)
