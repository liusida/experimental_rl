import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erl", # Replace with your own username
    version="0.0.1",
    author="Sida Liu",
    author_email="learner.sida.liu@gmail.com",
    description="Experimental RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liusida/experimental_rl",
    project_urls={
        "Bug Tracker": "https://github.com/liusida/experimental_rl/issues",
    },
    py_modules=['erl'],
    python_requires=">=3.8",
)