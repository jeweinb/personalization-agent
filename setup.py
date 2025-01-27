from setuptools import setup, find_packages

setup(name='personalization-agent',
      version='2.0',
      description='personalization reinforcement learning agent',
      author='Jason Weinberg',
      author_email='xxxx',
      python_requires='>=3.8',
      packages=find_packages(include=['agent', 'agent.*']),
      install_requires=[
            'numpy==1.23.5',
            'pandas',
            'scipy',
            'requests',
            'gymnasium',
            'ray[default]==2.1.0',
            'ray[rllib]',
            'ray[serve]',
            'ray[tune]',
            'optuna',
            'mlflow',
            'torch==2.0.1',
            'pytorch-lightning==1.8.4',
            'scikit-learn',
            'joblib',
            'tqdm',
            'seaborn',
            'tensorboard',
            'pyarrow',
            'pyarrow',
            'pyyaml',
            'azure-core',
            'azure-keyvault-secrets',
            'azure-identity',
            'azure-ai-ml',
            'azureml-mlflow',
            'azureml-inference-server-http',
            'databricks-sql-connector',
      ],
)
