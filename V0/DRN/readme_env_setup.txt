
Virtual env configuration:

-----------

# assume python3.6 is installed

sudo apt-get install python3.6-dev  # required to compile PyICU and pycld2

sudo apt-get install python3.6-venv
python3.6 -m venv venv  # create venv folder inside
source venv/bin/activate  # get inside env


pip install -r requirements.txt
# morfessor==2.0.3 is added instead of gensim==3.3.0 or gensim[test]

# before the first run, comment lines 35-38 in venv/lib/python3.6/site-packages/pattern3/text/tree.py

python download_polyglot.py

# now ready to run scripts
python test1.py