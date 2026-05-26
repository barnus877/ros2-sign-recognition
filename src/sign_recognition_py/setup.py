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
        ('share/' + package_name + '/network_model',
            ['network_model/model.best.keras',
             'network_model/model.keras',
             'network_model/model_training.png']),
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
            'save_training_images = sign_recognition_py.save_training_images:main',
            'line_follower = sign_recognition_py.line_follower:main',
            'line_follower_cnn = sign_recognition_py.line_follower_cnn:main',
            'sign_recogniser = sign_recognition_py.sign_recogniser_cnn:main',
        ],
    },
)
