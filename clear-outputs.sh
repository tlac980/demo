#!/usr/bin/env bash

jupyter nbconvert demo.ipynb --to notebook --ClearOutputPreprocessor.enabled=True --output demo

pip uninstall scikit-ika
