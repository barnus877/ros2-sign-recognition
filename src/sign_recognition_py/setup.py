from setuptools import find_packages, setup

package_name = 'sign_recognition_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saserq',
    maintainer_email='sas.mia.mor@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'helloka_py = sign_recognition_py.hello_py:main',
            'save_training_images = sign_recognition_py.save_training_images:main',
        ],
    },
)
