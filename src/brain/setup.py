from setuptools import find_packages, setup

package_name = 'brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install all .py files in the launch directory
        ('share/' + package_name + '/launch', [
            'launch/brain_launch.py', 
            'launch/system_launch.py', 
            'launch/project_launch.py'
        ]), 
    ],
    install_requires=['setuptools', 'key_pos_msgs'],
    zip_safe=True,
    maintainer='mtrn',
    maintainer_email='mtrn@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brain = brain.brain:main',
            'talker_b = brain.moveit_pub:main',  
        ],
    },
)