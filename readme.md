**CAP5610 Diabetes Classification**

*Kyle Merrit, Adam Sachs, Ryan Harding, Miguel Colmenares, Rodrigo Vena*

*Note on ipynb files*

Any ipynb files left in this repository were used during the development of this project, and as such are included for a look into the development process. They are not part of the main program, and should not be run to test final output, but can be inspected and run to replicate our development process.

*Required Libraries*
This program heavily implements libraries such as sklearn, imblern, torch, transformers, numpy, matplotlib, and pandas. Many of these should be installed by default, and the following command should install any that aren't. If a library is missing, simply install using pip on the environment that you're running from.

    'python3 -m pip install transformers torch scikit-learn pandas imbalanced-learn'

*How To Run*

Using a simple menu script, much of this program has been automated for convenience. Simply run the following from a terminal:

    'python3 main.py'

And you will be presented with a menu asking for your selection of model. Each model may have more specific options that the program will ask you about after, which are outlined in full in the PDF report under the 'Tutorial' subsection for every model.

For every model, after selecting those options, a fitting and three predictions will run, on the training, validation, and testing sets. After completion, the program will output Confusion Matrices to a new window, and the classification reports to the terminal. The program will terminate after all matrices and reports have been displayed, and can be run again to test a different model.