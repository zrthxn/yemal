# Machine Learning
Introduction to the hot topics

## About
This repository is a sort of storage for basic ML programs 
that I do. Each project is in a separate folder and mostly 
these will be Jupyter notebooks because of the ease.

So if someone wants to use this, I have provided a shell script for UNIX systems that will download and setup the datasets for you. All you have to do is
```
# Enter the project directory
$ cd ~/yemal

# Give permission to execute the shell script
$ sudo chmod +x download.sh 

#Execute
$ ./download.sh
```
Otherwise there is a dataset download link in the readme in most projects. There is a trained model pickle included wherever possible.


# Instructions
The following instructions will get you up to speed on how to install everything that is necessary. Please be advised, installing all these by hand is a very risky game. If you're lucky you'll be able to figure it out in about an hour. If not, you'll spend days pulling your hair out.

## Steps to Start
1. Install Python 3.6.0 and only **Python 3.6.0** from the official python website. This will be the root python interpreter for your system.

2. Go to Anaconda's website and download the Anaconda setup for your system. Anaconda, or more precisely it's package manager **conda** will help you manage all the millions of lines of code.

3. Once Anaconda is installed, add the install path and the Scripts directory to your PATH system environment variable.

4. You should now be able to access *conda* from your terminal
  ```
  $ conda --version
  ```

5. Create a Virtual Environment with conda. Here you can choose whatever venv name you want. Keep the Python version at 3.6 to avoid errors.
  ```
  $ conda create -n <venv> python=3.6
  ```

6. Activate the venv by
  ```
  $ conda activate <venv>
  # You're now in conda world!
  ```

7. Use pip to install any packages you want. We will install the requirements from our requirements.txt file
  ```
  (venv) $ cd ~/yemal
  (venv) $ pip install -r requirements.txt
  ```


### Required
- Anaconda 3
- Python 3.6

### Recommended
- Jupyter
- CUDA Toolkit

