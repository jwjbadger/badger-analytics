# Badger Analytics

## Introduction

This is a package that is built to do basic data analysis from scratch. This package is meant to give a large amount of customizability in each data analysis method.

## Importing Data

There are two main methods of importing external data.

### Local

If you want to import a file that is on your machine, you can use the function `data_import.local_import(path)`

This function takes a path as an input. This can be a relative or absolute path. Data should be formatted using standard python array format (i.e. `[[1,2,3],[4,5,6],[7,8,9]]`)

This function returns a [NumPy](https://NumPy.org/ "NumPy home page") array.

### Remote

If you want to import a file via HTTP get request, you can use the `data_import.remote_import(path)` function

This function is exactly the same as `data_import.local_import()` except the path it takes as input should be a website. Be sure to include "http://" or "https://" in front of the path.

## Data Processing

Before any analytical proceses can be performed, the data may have to be cleaned up.

## Standardization

The goal of standardization is to compress data into a specific range.

### Z Score

This method of standardization is accessible via `processing.standardization.z_score(input_set)`

Given a 2D NumPy array with more than 1 row, this function will return the z scores of the inputed data set's numerical values. The result will be a NumPy array.
