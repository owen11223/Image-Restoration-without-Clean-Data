import wget
import tarfile
#pip install python3-wget

bsd300_dir = 'imgs/bsd300_dir'
url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz'
name = wget.download(url)
my_tar = tarfile.open(name)
my_tar.extractall('imgs') # specify which folder to extract to
my_tar.close()
