from setuptools import find_packages, setup

package_name = "robot_api"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/moveit_franka_python.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "manipulation_example = robot_api.manipulation_example:main",
            "manipulation = robot_api.manipulation:main",
            "task_manager = robot_api.task_manager:main",
        ],
    },
)
