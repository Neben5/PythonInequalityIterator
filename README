# Python Inequality System

## Description

Assists in the visualization of systems of inequalities on hemihedra

## Usage

* Activate the conda environment

```shell
# If first time
$ conda env create -f environment.yml
# Always
$ conda activate PythonInequalitySystem
```

* Create a python notebook ___in the same directory___ as operations.py and import it.
* Use parse_system(filename: string) to parse your input system and get the initial state and equations (inequalities)
* Graph your system using plot_hemi(hemihedra). Combining multiple plots in a single cell will render them all in the same graph.
* Iterate states with apply_system(previous_state, equations)

### Input Specification

The format of inequality files will be as follows:

> There are three valid types of lines: Comments, Initial Conditions, and Inequalities

#### Comments

Denoted by a '%' as the first character of the line

`% this is a comment`

#### Initial Conditions

This will be a set of points that a variable contains at its initial state. The format is as follows

`X >= {(0.0,0.0),(1.1,1.1)...(9.9,9.9)}`

Note the single character variable name.

Note the spaces before and after the inequality sign, and the lack of them elsewhere.

Note that all numbers are denoted by [0-9]+.[0-9]* in regex. Not having a decimal point, or a digit to either side of it, will cause an error.

#### Inequalities

This will denote that the LHS contains the RHS based on a normalized minkowski sum.


`X >= (0.5)X + (0.5)Y`

Again, whitespace is key, and so is the formatting of numbers and variables.

Aditionally, the parentheses around the coefficients of variables are required.

## Examples

The examples folder contains both example inputs and notebooks.
Note that these will need to be in the same directory together to work out of the box.

## Limitations and Acknowledgements

Acknowledgements to come

Inequalities currently must have two (2) elements. Adding more complex statements is currently in progress and not supported. At all.