from setuptools import find_packages, setup

package_name = 'key_pub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/key_pub_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='finn',
    maintainer_email='z5309129@ad.unsw.edu.au',
    description='TODO: Package description',
    license='Apache-2.0',
    # Comment out if it causes issues
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = key_pub.key_publisher:main',
            'dummy_talker = key_pub.dummy_key_publisher:main',
        ],
    },
)